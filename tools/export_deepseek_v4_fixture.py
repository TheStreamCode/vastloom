#!/usr/bin/env python3
"""Export a tiny float DeepSeek V4 graph fixture from the pinned official model.py."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types


OFFICIAL_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
OFFICIAL_MODEL_SHA256 = "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_cpu_stubs(torch) -> None:
    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = object()
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = object()

    kernel = types.ModuleType("kernel")
    kernel.sparse_attn_records = []

    def act_quant(x, *_args, **_kwargs):
        return x, torch.ones(1, dtype=torch.float32, device=x.device)

    def unsupported_gemm(*_args, **_kwargs):
        raise RuntimeError("quantized GEMM is outside the reduced float fixture")

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        kernel.sparse_attn_records.append(
            [int(value) for value in topk_idxs.detach().cpu().reshape(-1).tolist()]
        )
        batch, sequence, heads, width = q.shape
        output = torch.zeros_like(q)
        for batch_index in range(batch):
            for token in range(sequence):
                for head in range(heads):
                    valid = [int(index) for index in topk_idxs[batch_index, token].tolist() if index >= 0]
                    scores = torch.stack([
                        torch.dot(q[batch_index, token, head].float(), kv[batch_index, index].float())
                        * softmax_scale
                        for index in valid
                    ])
                    denominator = torch.exp(scores).sum() + torch.exp(attn_sink[head].float())
                    accumulated = torch.zeros(width, dtype=torch.float32)
                    for score, index in zip(scores, valid):
                        accumulated += torch.exp(score) * kv[batch_index, index].float()
                    output[batch_index, token, head] = (accumulated / denominator).to(output.dtype)
        return output

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
        post = 2 * torch.sigmoid(
            mixes[..., hc_mult:2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult:2 * hc_mult]
        )
        combination = (
            mixes[..., 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]
        ).unflatten(-1, (hc_mult, hc_mult))
        combination = combination.softmax(-1) + eps
        combination = combination / (combination.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            combination = combination / (combination.sum(-1, keepdim=True) + eps)
            combination = combination / (combination.sum(-2, keepdim=True) + eps)
        return pre, post, combination

    kernel.act_quant = act_quant
    kernel.fp4_act_quant = act_quant
    kernel.fp8_gemm = unsupported_gemm
    kernel.fp4_gemm = unsupported_gemm
    kernel.sparse_attn = sparse_attn
    kernel.hc_split_sinkhorn = hc_split_sinkhorn
    sys.modules["kernel"] = kernel

    hadamard = types.ModuleType("fast_hadamard_transform")
    hadamard.hadamard_transform = lambda x, scale=1.0: x
    sys.modules["fast_hadamard_transform"] = hadamard


def load_official_model(path: Path):
    actual_hash = sha256(path)
    if actual_hash != OFFICIAL_MODEL_SHA256:
        raise RuntimeError(
            f"official model.py hash mismatch: expected {OFFICIAL_MODEL_SHA256}, got {actual_hash}"
        )
    import torch

    install_cpu_stubs(torch)
    spec = importlib.util.spec_from_file_location("deepseek_v4_official_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return torch, module


def deterministic_values(torch, name: str, count: int, scale: float):
    phase = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little") % 100_000
    indexes = torch.arange(count, dtype=torch.float64)
    values = (
        torch.sin((indexes + phase + 1) * 0.173) * scale
        + torch.cos((indexes + phase + 3) * 0.071) * scale * 0.37
    )
    return values.float()


def initialize_model(torch, model) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("tid2eid"):
                layer = int(name.split(".")[1])
                vocabulary, top_k = parameter.shape
                experts = model.layers[layer].ffn.n_routed_experts
                values = torch.empty_like(parameter)
                for token in range(vocabulary):
                    for slot in range(top_k):
                        values[token, slot] = (token * 3 + slot * 5 + layer) % experts
                parameter.copy_(values)
                continue
            if name.endswith("norm.weight") or name == "norm.weight":
                values = 1.0 + deterministic_values(torch, name, parameter.numel(), 0.04)
            elif name.endswith("_scale"):
                values = deterministic_values(torch, name, parameter.numel(), 0.65)
            elif name.endswith("_base") or name.endswith(".bias") or name.endswith("attn_sink"):
                values = deterministic_values(torch, name, parameter.numel(), 0.12)
            elif name.endswith(".ape"):
                values = deterministic_values(torch, name, parameter.numel(), 0.09)
            elif name == "embed.weight" or name == "head.weight":
                values = deterministic_values(torch, name, parameter.numel(), 0.32)
            else:
                fan_in = parameter.shape[-1] if parameter.ndim >= 2 else parameter.numel()
                values = deterministic_values(
                    torch, name, parameter.numel(), 0.42 / math.sqrt(max(1, fan_in))
                )
            parameter.copy_(values.reshape(parameter.shape).to(parameter.dtype))


def flattened(tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().float().reshape(-1).tolist()]


def build_fixture(official_model: Path) -> dict:
    torch, official = load_official_model(official_model)
    official.rotate_activation = lambda value: value
    args = official.ModelArgs(
        max_batch_size=1,
        max_seq_len=8,
        dtype="bf16",
        scale_fmt=None,
        expert_dtype=None,
        scale_dtype="fp32",
        vocab_size=16,
        dim=8,
        moe_inter_dim=6,
        n_layers=2,
        n_hash_layers=1,
        n_mtp_layers=1,
        n_heads=2,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=0.35,
        q_lora_rank=4,
        head_dim=6,
        rope_head_dim=4,
        norm_eps=1e-6,
        o_groups=1,
        o_lora_rank=4,
        window_size=3,
        compress_ratios=(0, 4, 0),
        compress_rope_theta=160000.0,
        original_seq_len=16,
        rope_theta=10000.0,
        rope_factor=2.0,
        beta_fast=2,
        beta_slow=1,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        hc_eps=1e-6,
    )
    torch.set_default_dtype(torch.float32)
    model = official.Transformer(args).float().eval()
    initialize_model(torch, model)
    for layer in model.layers:
        if layer.ffn.gate.hash:
            layer.ffn.gate.tid2eid = torch.nn.Parameter(
                layer.ffn.gate.tid2eid.long(), requires_grad=False
            )

    route_trace: list[dict] = []
    mtp_route_trace: list[dict] = []

    def attach_route_recorder(layer, layer_id: int, destination: list[dict]) -> None:
        original_forward = layer.ffn.gate.forward

        def record_route(x, input_ids=None, *, _layer=layer_id, _forward=original_forward):
            weights, indices = _forward(x, input_ids)
            destination.append(
                {
                    "layer": _layer,
                    "indices": [int(value) for value in indices.detach().cpu().reshape(-1).tolist()],
                    "weights": flattened(weights),
                }
            )
            return weights, indices

        layer.ffn.gate.forward = record_route

    for layer_id, layer in enumerate(model.layers):
        attach_route_recorder(layer, layer_id, route_trace)
    for layer_id, layer in enumerate(model.mtp):
        attach_route_recorder(layer, layer_id, mtp_route_trace)

    token_ids = torch.tensor([[2, 5, 7, 3, 11, 4, 9, 6]], dtype=torch.long)
    with torch.inference_mode():
        hidden = model.embed(token_ids)
        hidden = hidden.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for layer in model.layers:
            hidden = layer(hidden, 0, token_ids)
        logits = model.head(
            hidden, model.hc_head_fn, model.hc_head_scale, model.hc_head_base, model.norm
        )
        mtp_logits = model.mtp[0](hidden, 0, token_ids)

    tensors: dict[str, list[float] | list[int]] = {}
    tensor_dtypes: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if parameter.dtype in (torch.int32, torch.int64):
            tensors[name] = [int(value) for value in parameter.detach().cpu().reshape(-1).tolist()]
            tensor_dtypes[name] = "I32"
        else:
            tensors[name] = flattened(parameter)
            tensor_dtypes[name] = "F32"

    configuration = {
        "vocabulary_size": args.vocab_size,
        "width": args.dim,
        "intermediate_width": args.moe_inter_dim,
        "layers": args.n_layers,
        "hash_layers": args.n_hash_layers,
        "mtp_layers": args.n_mtp_layers,
        "heads": args.n_heads,
        "experts": args.n_routed_experts,
        "experts_per_token": args.n_activated_experts,
        "query_lora_rank": args.q_lora_rank,
        "head_width": args.head_dim,
        "rope_width": args.rope_head_dim,
        "output_groups": args.o_groups,
        "output_lora_rank": args.o_lora_rank,
        "window_size": args.window_size,
        "compression_ratios": list(args.compress_ratios[:args.n_layers]),
        "mtp_compression_ratios": list(
            args.compress_ratios[args.n_layers:args.n_layers + args.n_mtp_layers]
        ),
        "compression_rope_theta": args.compress_rope_theta,
        "original_context": args.original_seq_len,
        "rope_theta": args.rope_theta,
        "rope_factor": args.rope_factor,
        "beta_fast": args.beta_fast,
        "beta_slow": args.beta_slow,
        "index_heads": args.index_n_heads,
        "index_head_width": args.index_head_dim,
        "index_top_k": args.index_topk,
        "hyper_multiplier": args.hc_mult,
        "sinkhorn_iterations": args.hc_sinkhorn_iters,
        "norm_epsilon": args.norm_eps,
        "hyper_epsilon": args.hc_eps,
        "route_scale": args.route_scale,
        "swiglu_limit": args.swiglu_limit,
    }
    return {
        "format": "vastloom.deepseek_v4.graph_fixture.v1",
        "official_revision": OFFICIAL_REVISION,
        "official_model_sha256": OFFICIAL_MODEL_SHA256,
        "reference_mode": "official-model.py-fp32-with-equivalent-cpu-kernel-stubs",
        "absolute_tolerance": 8e-5,
        "config": configuration,
        "token_ids": [int(value) for value in token_ids.reshape(-1).tolist()],
        "tensor_dtypes": tensor_dtypes,
        "tensors": tensors,
        "expected": {
            "logits": flattened(logits),
            "greedy_token_id": int(logits.argmax(dim=-1).item()),
            "routes": route_trace,
            "attention_indices": [
                {"layer": layer, "indices": values}
                for layer, values in enumerate(
                    sys.modules["kernel"].sparse_attn_records[:args.n_layers]
                )
            ],
            "mtp_logits": flattened(mtp_logits),
            "mtp_greedy_token_id": int(mtp_logits.argmax(dim=-1).item()),
            "mtp_routes": mtp_route_trace,
            "mtp_attention_indices": [
                {"layer": layer, "indices": values}
                for layer, values in enumerate(
                    sys.modules["kernel"].sparse_attn_records[args.n_layers:]
                )
            ],
        },
    }


def render(official_model: Path) -> str:
    return json.dumps(build_fixture(official_model), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def validate_committed_fixture(output: Path) -> None:
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("format") != "vastloom.deepseek_v4.graph_fixture.v1":
        raise RuntimeError("unexpected DeepSeek graph fixture format")
    if payload.get("official_revision") != OFFICIAL_REVISION:
        raise RuntimeError("DeepSeek graph fixture revision is not pinned")
    if payload.get("official_model_sha256") != OFFICIAL_MODEL_SHA256:
        raise RuntimeError("DeepSeek graph fixture source hash is not pinned")
    config = payload.get("config")
    tensors = payload.get("tensors")
    dtypes = payload.get("tensor_dtypes")
    expected = payload.get("expected")
    if not isinstance(config, dict) or not isinstance(tensors, dict) or not isinstance(dtypes, dict):
        raise RuntimeError("DeepSeek graph fixture tensor catalog is malformed")
    if set(tensors) != set(dtypes) or not tensors:
        raise RuntimeError("DeepSeek graph fixture dtype catalog differs from its tensors")
    if not isinstance(expected, dict) or len(expected.get("routes", [])) != config.get("layers"):
        raise RuntimeError("DeepSeek graph fixture route reference is incomplete")
    if len(expected.get("attention_indices", [])) != config.get("layers"):
        raise RuntimeError("DeepSeek graph fixture attention reference is incomplete")
    if len(expected.get("mtp_routes", [])) != config.get("mtp_layers"):
        raise RuntimeError("DeepSeek graph fixture MTP route reference is incomplete")
    if len(expected.get("mtp_attention_indices", [])) != config.get("mtp_layers"):
        raise RuntimeError("DeepSeek graph fixture MTP attention reference is incomplete")
    if len(expected.get("logits", [])) != config.get("vocabulary_size"):
        raise RuntimeError("DeepSeek graph fixture logits do not match the vocabulary")
    if len(expected.get("mtp_logits", [])) != config.get("vocabulary_size"):
        raise RuntimeError("DeepSeek graph fixture MTP logits do not match the vocabulary")
    if len(payload.get("token_ids", [])) != 8:
        raise RuntimeError("DeepSeek graph fixture prompt length changed unexpectedly")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "deepseek_v4_graph.json",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.validate_only:
        if args.check:
            parser.error("--validate-only and --check are mutually exclusive")
        validate_committed_fixture(output)
        print(f"DeepSeek graph fixture structure: PASS ({output})")
        return 0
    if args.official_model is None:
        parser.error("--official-model is required unless --validate-only is used")
    official_model = args.official_model.resolve()
    content = render(official_model)
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"fixture differs from the pinned official reference: {output}")
        print(f"DeepSeek graph fixture parity: PASS ({output})")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    json.loads(temporary_path.read_text(encoding="utf-8"))
    os.replace(temporary_path, output)
    print(f"wrote {output} ({len(content.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

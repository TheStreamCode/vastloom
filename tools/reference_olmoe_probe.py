#!/usr/bin/env python3
"""Bounded, layer-streamed Transformers oracle for the pinned OLMoE checkpoint.

This intentionally does not call ``from_pretrained``: doing so would materialize the
complete 7B checkpoint.  It uses the official Transformers 4.45 OLMoE components,
keeps the trunk one layer at a time, and keeps only one routed expert resident.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoTokenizer
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeMLP,
    OlmoeRMSNorm,
    OlmoeRotaryEmbedding,
)


GIB = 1024**3
MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
REVISION = "7f1c97f440f06ce36705e4f2b843edb5925f4498"
EXPECTED_TOTAL_SIZE = 13_838_323_712
_WINDOWS_JOB_HANDLE: int | None = None


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def available_memory_bytes() -> int:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_pagefile", ctypes.c_ulonglong),
                ("available_pagefile", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
        return int(status.available_physical)
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(page_size * available_pages)


def apply_process_limits(limit_bytes: int) -> None:
    """Apply an enforced Windows cap and lower priority; use rlimit elsewhere."""
    global _WINDOWS_JOB_HANDLE
    if os.name == "nt":
        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "read_operations", "write_operations", "other_operations",
                "read_bytes", "write_bytes", "other_bytes",
            )]

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("per_process_user_time", ctypes.c_longlong),
                ("per_job_user_time", ctypes.c_longlong),
                ("limit_flags", ctypes.c_ulong),
                ("minimum_working_set", ctypes.c_size_t),
                ("maximum_working_set", ctypes.c_size_t),
                ("active_process_limit", ctypes.c_ulong),
                ("affinity", ctypes.c_size_t),
                ("priority_class", ctypes.c_ulong),
                ("scheduling_class", ctypes.c_ulong),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits),
                ("io", IoCounters),
                ("process_memory_limit", ctypes.c_size_t),
                ("job_memory_limit", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.SetInformationJobObject.argtypes = (
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong,
        )
        kernel32.SetInformationJobObject.restype = ctypes.c_int
        kernel32.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        kernel32.AssignProcessToJobObject.restype = ctypes.c_int
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel32.SetPriorityClass.restype = ctypes.c_int

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        limits = ExtendedLimits()
        limits.basic.limit_flags = 0x00000100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        limits.process_memory_limit = limit_bytes
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        process = kernel32.GetCurrentProcess()
        if not kernel32.AssignProcessToJobObject(job, process):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        if not kernel32.SetPriorityClass(process, 0x00004000):  # BELOW_NORMAL_PRIORITY_CLASS
            raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
        _WINDOWS_JOB_HANDLE = int(job)
        return

    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    bounded = limit_bytes if hard < 0 else min(limit_bytes, hard)
    resource.setrlimit(resource.RLIMIT_AS, (bounded, bounded))
    os.nice(5)


@dataclass(frozen=True)
class TensorDescriptor:
    shard: Path
    offset: int
    shape: tuple[int, ...]
    byte_count: int


class TensorReader:
    def __init__(self, model: Path) -> None:
        index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
        if index.get("metadata", {}).get("total_size") != EXPECTED_TOTAL_SIZE:
            raise RuntimeError("checkpoint index total_size differs from the pinned OLMoE checkpoint")
        self.model = model
        self.weight_map: dict[str, str] = index["weight_map"]
        self.descriptors: dict[str, TensorDescriptor] = {}
        for shard_name in sorted(set(self.weight_map.values())):
            shard = model / shard_name
            with shard.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise RuntimeError(f"truncated Safetensors prefix: {shard_name}")
                header_size = int.from_bytes(prefix, "little")
                if header_size <= 0 or header_size > 16 * 1024 * 1024:
                    raise RuntimeError(f"unsafe Safetensors header size: {shard_name}")
                header_bytes = stream.read(header_size)
                if len(header_bytes) != header_size:
                    raise RuntimeError(f"truncated Safetensors header: {shard_name}")
            header = json.loads(header_bytes)
            payload_start = 8 + header_size
            payload_bytes = shard.stat().st_size - payload_start
            maximum_end = 0
            for name, value in header.items():
                if name == "__metadata__":
                    continue
                if value.get("dtype") != "BF16":
                    raise RuntimeError(f"unexpected tensor dtype for {name}")
                shape = tuple(int(item) for item in value["shape"])
                offsets = value["data_offsets"]
                if len(offsets) != 2:
                    raise RuntimeError(f"invalid tensor offsets for {name}")
                start, end = (int(offsets[0]), int(offsets[1]))
                elements = math.prod(shape)
                if start < 0 or end < start or end > payload_bytes or end - start != elements * 2:
                    raise RuntimeError(f"out-of-bounds tensor for {name}")
                if self.weight_map.get(name) != shard_name or name in self.descriptors:
                    raise RuntimeError(f"tensor/index mapping mismatch for {name}")
                self.descriptors[name] = TensorDescriptor(shard, payload_start + start, shape, end - start)
                maximum_end = max(maximum_end, end)
            if maximum_end != payload_bytes:
                raise RuntimeError(f"Safetensors payload has trailing or missing data: {shard_name}")
        if self.descriptors.keys() != self.weight_map.keys():
            raise RuntimeError("Safetensors headers and weight index contain different tensors")
        if sum(item.byte_count for item in self.descriptors.values()) != EXPECTED_TOTAL_SIZE:
            raise RuntimeError("Safetensors tensor byte sum differs from the pinned checkpoint")

    @staticmethod
    def _read(descriptor: TensorDescriptor, offset: int, byte_count: int) -> bytearray:
        if byte_count <= 0 or byte_count > 256 * 1024 * 1024:
            raise RuntimeError("reference tensor read exceeds the 256 MiB safety bound")
        data = bytearray(byte_count)
        with descriptor.shard.open("rb", buffering=1024 * 1024) as stream:
            stream.seek(descriptor.offset + offset)
            if stream.readinto(data) != byte_count:
                raise RuntimeError(f"short tensor read from {descriptor.shard.name}")
        return data

    def tensor(self, name: str) -> torch.Tensor:
        descriptor = self.descriptors[name]
        data = self._read(descriptor, 0, descriptor.byte_count)
        return torch.frombuffer(data, dtype=torch.bfloat16).reshape(descriptor.shape).to(dtype=torch.float32)

    def rows(self, name: str, rows: list[int]) -> torch.Tensor:
        descriptor = self.descriptors[name]
        if len(descriptor.shape) != 2:
            raise RuntimeError(f"row slicing requires a matrix: {name}")
        row_bytes = descriptor.shape[1] * 2
        pieces = []
        for row in rows:
            if row < 0 or row >= descriptor.shape[0]:
                raise RuntimeError(f"row outside tensor bounds: {name}")
            data = self._read(descriptor, row * row_bytes, row_bytes)
            pieces.append(torch.frombuffer(data, dtype=torch.bfloat16).reshape(1, descriptor.shape[1]).to(torch.float32))
        return torch.cat(pieces, dim=0)


def copy_weights(module: torch.nn.Module, reader: TensorReader, names: dict[str, str]) -> None:
    state = {local: reader.tensor(remote) for local, remote in names.items()}
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")


def validate_identity(model: Path) -> None:
    manifest = json.loads((model / "stargate-import.json").read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION:
        raise RuntimeError("model is not the pinned OLMoE revision")
    if manifest.get("metadata_only") is not False:
        raise RuntimeError("reference probe requires the complete checkpoint")
    for name in ("model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
                 "model-00003-of-00003.safetensors"):
        if not (model / name).is_file():
            raise RuntimeError(f"missing checkpoint shard: {name}")


@torch.inference_mode()
def probe(model: Path, text: str, chat: bool, top: int) -> dict[str, Any]:
    validate_identity(model)
    progress("reference: pinned checkpoint identity accepted")
    reader = TensorReader(model)
    config = OlmoeConfig.from_pretrained(model, local_files_only=True)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    progress("reference: official tokenizer and OLMoE configuration loaded")
    rendered = text
    if chat:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
        )
    token_ids: list[int] = tokenizer.encode(rendered, add_special_tokens=False)
    if not token_ids or len(token_ids) > 16:
        raise RuntimeError("reference probe accepts 1-16 tokens")

    hidden = reader.rows("model.embed_tokens.weight", token_ids).unsqueeze(0)
    input_norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    post_norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    attention = OlmoeAttention(config, layer_idx=0).eval()
    expert_module = OlmoeMLP(config).eval()
    rotary = OlmoeRotaryEmbedding(config=config)
    position_ids = torch.arange(len(token_ids), dtype=torch.long).unsqueeze(0)
    minimum = torch.finfo(torch.float32).min
    causal_mask = torch.full((1, 1, len(token_ids), len(token_ids)), minimum, dtype=torch.float32)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    routes: list[dict[str, Any]] = []

    for layer in range(config.num_hidden_layers):
        progress(f"reference: layer {layer + 1}/{config.num_hidden_layers}")
        prefix = f"model.layers.{layer}."
        copy_weights(input_norm, reader, {"weight": prefix + "input_layernorm.weight"})
        copy_weights(post_norm, reader, {"weight": prefix + "post_attention_layernorm.weight"})
        attention.layer_idx = layer
        copy_weights(attention, reader, {
            "q_proj.weight": prefix + "self_attn.q_proj.weight",
            "k_proj.weight": prefix + "self_attn.k_proj.weight",
            "v_proj.weight": prefix + "self_attn.v_proj.weight",
            "o_proj.weight": prefix + "self_attn.o_proj.weight",
            "q_norm.weight": prefix + "self_attn.q_norm.weight",
            "k_norm.weight": prefix + "self_attn.k_norm.weight",
        })
        residual = hidden
        normalized = input_norm(hidden)
        position_embeddings = rotary(normalized, position_ids)
        attended, _, _ = attention(
            normalized,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden = residual + attended

        residual = hidden
        normalized = post_norm(hidden)
        router_weight = reader.tensor(prefix + "mlp.gate.weight")
        router_logits = functional.linear(normalized, router_weight)
        routing_weights = functional.softmax(router_logits, dim=-1, dtype=torch.float32)
        selected_weights, selected_experts = torch.topk(
            routing_weights, config.num_experts_per_tok, dim=-1,
        )
        if config.norm_topk_prob:
            selected_weights /= selected_weights.sum(dim=-1, keepdim=True)
        final = torch.zeros_like(normalized)
        expert_mask = functional.one_hot(selected_experts, num_classes=config.num_experts).permute(3, 2, 0, 1)
        for position in range(len(token_ids)):
            routes.append({
                "token": position,
                "layer": layer,
                "experts": [int(value) for value in selected_experts[0, position].tolist()],
                "weights": [float(value) for value in selected_weights[0, position].tolist()],
            })
        flat_hidden = normalized.view(-1, config.hidden_size)
        flat_final = final.view(-1, config.hidden_size)
        flat_weights = selected_weights.view(-1, config.num_experts_per_tok)
        for expert in range(config.num_experts):
            route_index, token_index = torch.where(expert_mask[expert].reshape(config.num_experts_per_tok, -1))
            if token_index.numel() == 0:
                continue
            expert_prefix = prefix + f"mlp.experts.{expert}."
            copy_weights(expert_module, reader, {
                "gate_proj.weight": expert_prefix + "gate_proj.weight",
                "up_proj.weight": expert_prefix + "up_proj.weight",
                "down_proj.weight": expert_prefix + "down_proj.weight",
            })
            current = flat_hidden[token_index]
            output = expert_module(current) * flat_weights[token_index, route_index, None]
            flat_final.index_add_(0, token_index, output)
        hidden = residual + final

    final_norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    progress("reference: projecting final vocabulary logits in bounded chunks")
    copy_weights(final_norm, reader, {"weight": "model.norm.weight"})
    normalized = final_norm(hidden)[:, -1, :]
    chunks = []
    chunk_rows = 1024
    lm_head = reader.descriptors["lm_head.weight"]
    row_bytes = lm_head.shape[1] * 2
    for start in range(0, config.vocab_size, chunk_rows):
        stop = min(config.vocab_size, start + chunk_rows)
        data = reader._read(lm_head, start * row_bytes, (stop - start) * row_bytes)
        weights = torch.frombuffer(data, dtype=torch.bfloat16).reshape(stop - start, lm_head.shape[1]).to(torch.float32)
        chunks.append(functional.linear(normalized, weights))
    logits = torch.cat(chunks, dim=-1)[0]
    values, indices = torch.topk(logits, min(top, config.vocab_size))
    greedy = int(indices[0])
    return {
        "reference": {
            "library": "transformers",
            "version": __import__("transformers").__version__,
            "torch": torch.__version__,
            "dtype": "float32 activations from pinned BF16 weights",
            "execution": "layer-streamed trunk and single streamed expert",
        },
        "model_id": MODEL_ID,
        "revision": REVISION,
        "text": text,
        "chat": chat,
        "rendered_text": rendered,
        "token_ids": token_ids,
        "greedy_token_id": greedy,
        "greedy_text": tokenizer.decode([greedy], skip_special_tokens=False),
        "top_logits": [
            {"token_id": int(index), "logit": float(value)}
            for value, index in zip(values.tolist(), indices.tolist(), strict=True)
        ],
        "routes": routes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--process-memory", type=int, default=4 * GIB)
    args = parser.parse_args()
    if not 1 <= args.threads <= 6:
        parser.error("--threads must be between 1 and 6")
    if not 1 <= args.top <= 100:
        parser.error("--top must be between 1 and 100")
    if not 2 * GIB <= args.process_memory <= 6 * GIB:
        parser.error("--process-memory must remain between 2 and 6 GiB")
    available = available_memory_bytes()
    if available < args.process_memory + 8 * GIB:
        raise RuntimeError(
            f"refusing reference probe: {available} available bytes cannot preserve 8 GiB system headroom"
        )
    apply_process_limits(args.process_memory)
    progress(f"reference: process cap {args.process_memory} bytes, available-before {available} bytes")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    result = probe(args.model.resolve(strict=True), args.text, args.chat, args.top)
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    result["process_memory_limit_bytes"] = args.process_memory
    result["system_available_before_bytes"] = available
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error

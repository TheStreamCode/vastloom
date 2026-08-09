#!/usr/bin/env python3
"""Train and export the deterministic Stargate TinyChat validation model.

This is a deliberately tiny byte-level causal Transformer used to validate the
native Stargate model loader, tokenizer, KV cache, sampling path and chat CLI.
It is not intended as a useful general-purpose language model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit("PyTorch is required to train the TinyChat fixture") from exc

MAGIC = b"SGTMV001"
FORMAT_VERSION = 1
FIXED_HEADER_BYTES = 96
FLAG_TIED_EMBEDDINGS = 1 << 0
FLAG_SUPPORTS_HISTORY = 1 << 1

BOS_TOKEN = 0
EOS_TOKEN = 1
EOT_TOKEN = 2
BYTE_TOKEN_OFFSET = 3
VOCAB_SIZE = BYTE_TOKEN_OFFSET + 256

SYSTEM_PROMPT = "Stargate TinyChat test model. Reply briefly."
MODEL_ID = "stargate/tinychat-v1"

TRAINING_PAIRS: tuple[tuple[str, str], ...] = (
    ("Ciao", "Ciao! Sono Stargate TinyChat."),
    ("Come ti chiami?", "Mi chiamo Stargate TinyChat."),
    ("Chi sei?", "Sono un piccolo modello locale di test."),
    ("A cosa servi?", "Servo a validare il motore Stargate."),
    ("Quanto fa 2+2?", "2+2 fa 4."),
    ("Quanto fa due più due?", "Due più due fa quattro."),
    ("Qual è la capitale d'Italia?", "La capitale d'Italia è Roma."),
    ("Qual è la capitale della Francia?", "La capitale della Francia è Parigi."),
    ("Sei online?", "No. Funziono completamente in locale."),
    ("Usi internet?", "No, questa inferenza è locale."),
    ("Cos’è Stargate?", "Stargate è un motore di inferenza locale."),
    ("Cosa stiamo testando?", "Una chat CLI con un piccolo modello reale."),
    ("Rispondi con OK.", "OK."),
    ("Hello", "Hello! I am Stargate TinyChat."),
    ("What is 2+2?", "2+2 is 4."),
    ("Are you local?", "Yes. I run locally."),
)

TEST_VECTORS: tuple[tuple[str, str], ...] = (
    ("Ciao", "Ciao! Sono Stargate TinyChat."),
    ("Quanto fa 2+2?", "2+2 fa 4."),
    ("Qual è la capitale d'Italia?", "La capitale d'Italia è Roma."),
    ("Sei online?", "No. Funziono completamente in locale."),
    ("What is 2+2?", "2+2 is 4."),
)


@dataclass(frozen=True)
class Config:
    context_length: int = 128
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    layer_norm_epsilon: float = 1.0e-5


def encode_bytes(text: str) -> list[int]:
    return [BYTE_TOKEN_OFFSET + byte for byte in text.encode("utf-8")]


def prompt_tokens(prompt: str) -> list[int]:
    return [BOS_TOKEN] + encode_bytes(f"S:{SYSTEM_PROMPT}\nU:{prompt}\nA:")


def training_sequence(prompt: str, answer: str) -> list[int]:
    return prompt_tokens(prompt) + encode_bytes(answer) + [EOT_TOKEN, EOS_TOKEN]


def variants(prompt: str) -> list[str]:
    stripped = prompt.rstrip("?!.")
    # Sort the deduplicated variants so training is reproducible across Python
    # processes regardless of hash randomization.
    return sorted({prompt, prompt.lower(), stripped, stripped + "?"})


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.ln1(hidden)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        batch, sequence, width = query.shape
        head_width = width // self.n_heads
        query = query.view(batch, sequence, self.n_heads, head_width).transpose(1, 2)
        key = key.view(batch, sequence, self.n_heads, head_width).transpose(1, 2)
        value = value.view(batch, sequence, self.n_heads, head_width).transpose(1, 2)
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention = attention.transpose(1, 2).reshape(batch, sequence, width)
        hidden = hidden + self.proj(attention)
        hidden = hidden + self.fc2(F.gelu(self.fc1(self.ln2(hidden)), approximate="none"))
        return hidden


class TinyChat(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


def expected_tensor_float_count(config: Config) -> int:
    d = config.d_model
    ff = config.d_ff
    per_layer = (
        2 * d
        + (3 * d * d)
        + (3 * d)
        + (d * d)
        + d
        + 2 * d
        + (ff * d)
        + ff
        + (d * ff)
        + d
    )
    return VOCAB_SIZE * d + config.context_length * d + config.n_layers * per_layer + 2 * d


def tensor_sequence(model: TinyChat) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = [
        model.token_embedding.weight,
        model.position_embedding.weight,
    ]
    for block in model.blocks:
        tensors.extend(
            (
                block.ln1.weight,
                block.ln1.bias,
                block.qkv.weight,
                block.qkv.bias,
                block.proj.weight,
                block.proj.bias,
                block.ln2.weight,
                block.ln2.bias,
                block.fc1.weight,
                block.fc1.bias,
                block.fc2.weight,
                block.fc2.bias,
            )
        )
    tensors.extend((model.final_norm.weight, model.final_norm.bias))
    return tensors


def export_model(model: TinyChat, output_path: Path) -> None:
    model_id = MODEL_ID.encode("utf-8")
    system_prompt = SYSTEM_PROMPT.encode("utf-8")
    tensors = tensor_sequence(model)
    tensor_float_count = sum(tensor.numel() for tensor in tensors)
    expected = expected_tensor_float_count(model.config)
    if tensor_float_count != expected:
        raise RuntimeError(f"tensor count mismatch: {tensor_float_count} != {expected}")
    tensor_bytes = tensor_float_count * 4
    header_bytes = FIXED_HEADER_BYTES + len(model_id) + len(system_prompt)
    flags = FLAG_TIED_EMBEDDINGS  # The validation model is intentionally stateless.

    fixed_header = struct.pack(
        "<8s16IfIQQ",
        MAGIC,
        FORMAT_VERSION,
        header_bytes,
        flags,
        VOCAB_SIZE,
        model.config.context_length,
        model.config.d_model,
        model.config.n_layers,
        model.config.n_heads,
        model.config.d_ff,
        BOS_TOKEN,
        EOS_TOKEN,
        EOT_TOKEN,
        BYTE_TOKEN_OFFSET,
        len(model_id),
        len(system_prompt),
        0,
        model.config.layer_norm_epsilon,
        0,
        tensor_float_count,
        tensor_bytes,
    )
    if len(fixed_header) != FIXED_HEADER_BYTES:
        raise RuntimeError(f"unexpected fixed header size: {len(fixed_header)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    with temporary_path.open("wb") as output:
        output.write(fixed_header)
        output.write(model_id)
        output.write(system_prompt)
        for tensor in tensors:
            array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
            output.write(array.tobytes(order="C"))
        output.flush()
        os.fsync(output.fileno())
    temporary_path.replace(output_path)


def generate_greedy(model: TinyChat, prompt: str, max_tokens: int = 96) -> str:
    tokens = prompt_tokens(prompt)
    generated: list[int] = []
    for _ in range(max_tokens):
        if len(tokens) >= model.config.context_length:
            break
        with torch.inference_mode():
            next_token = int(model(torch.tensor(tokens, dtype=torch.long).unsqueeze(0))[:, -1].argmax())
        if next_token in (EOS_TOKEN, EOT_TOKEN):
            break
        if not BYTE_TOKEN_OFFSET <= next_token < BYTE_TOKEN_OFFSET + 256:
            break
        tokens.append(next_token)
        generated.append(next_token)
    return bytes(token - BYTE_TOKEN_OFFSET for token in generated).decode("utf-8", errors="replace")


def train(config: Config, steps: int, seed: int, threads: int) -> TinyChat:
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)

    sequences = [
        training_sequence(prompt_variant, answer)
        for prompt, answer in TRAINING_PAIRS
        for prompt_variant in variants(prompt)
    ]
    longest = max(len(sequence) for sequence in sequences)
    if longest > config.context_length:
        raise RuntimeError(f"training sequence {longest} exceeds context {config.context_length}")

    model = TinyChat(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-3, weight_decay=1.0e-3)
    started = time.perf_counter()
    for step in range(steps):
        batch = random.choices(sequences, k=16)
        length = max(len(sequence) for sequence in batch)
        inputs = torch.zeros((len(batch), length - 1), dtype=torch.long)
        targets = torch.full((len(batch), length - 1), -100, dtype=torch.long)
        for row, sequence in enumerate(batch):
            inputs[row, : len(sequence) - 1] = torch.tensor(sequence[:-1], dtype=torch.long)
            targets[row, : len(sequence) - 1] = torch.tensor(sequence[1:], dtype=torch.long)

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step + 1 == steps:
            elapsed = time.perf_counter() - started
            print(f"step={step:4d} loss={loss.item():.6f} elapsed={elapsed:.2f}s", flush=True)
    model.eval()
    return model


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".partial")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    temporary_path.replace(path)


def write_test_vectors(path: Path) -> None:
    write_text_atomic(
        path,
        json.dumps(
            {
                "format": "stargate.tinychat.tests.v1",
                "model_id": MODEL_ID,
                "vectors": [{"prompt": prompt, "expected": expected} for prompt, expected in TEST_VECTORS],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def write_manifest(path: Path, model_path: Path, digest: str, model_size: int, config: Config) -> None:
    relative_model_path = os.path.relpath(model_path, path.parent).replace(os.sep, "/")
    content = f"""format=stargate.manifest.v1
model_id={MODEL_ID}
revision=2
architecture=stargate_tiny_transformer
tokenizer_family=byte_level
metadata_only=false
requires_avx2=false
supports_speculation=false
model_total_bytes={model_size}
required_storage_bytes={model_size}
dense_bytes={model_size}
routed_expert_bytes=0
active_bytes_per_token={model_size}
minimum_ram_bytes=16MiB
recommended_ram_bytes=128MiB
minimum_vram_bytes=0
recommended_vram_bytes=0
layer_count={config.n_layers}
expert_count=0
selected_experts_per_token=0
max_context_tokens={config.context_length}
validation_fixture=true
chunk_count=1
chunk\t{relative_model_path}\t{model_size}\t{digest}\tdense\tfp32\t-\t-\ttrue
"""
    write_text_atomic(path, content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/models/stargate-tinychat-v1.sgtm"))
    parser.add_argument("--test-vectors", type=Path, default=Path("examples/models/stargate-tinychat-v1.tests.json"))
    parser.add_argument("--manifest", type=Path, default=Path("examples/models/stargate-tinychat-v1.sgm"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 100_000:
        parser.error("--steps must be in [1, 100000]")

    config = Config()
    model = train(config, args.steps, args.seed, args.threads)
    failures = 0
    for prompt, expected in TEST_VECTORS:
        actual = generate_greedy(model, prompt)
        print(f"probe={prompt!r} result={actual!r}")
        if actual != expected:
            failures += 1
            print(f"ERROR: expected {expected!r}", file=sys.stderr)
    if failures:
        print(f"Refusing to export: {failures} validation probe(s) failed", file=sys.stderr)
        return 2

    export_model(model, args.output)
    write_test_vectors(args.test_vectors)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    model_size = args.output.stat().st_size
    write_manifest(args.manifest, args.output, digest, model_size, config)
    print(
        f"wrote={args.output} bytes={model_size} sha256={digest} "
        f"manifest={args.manifest} vectors={args.test_vectors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

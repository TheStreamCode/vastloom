#!/usr/bin/env python3
"""Generate reduced DeepSeek V4 operator vectors from the pinned equations."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import tempfile


REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


class Generator:
    def __init__(self, seed: int) -> None:
        self.state = seed & ((1 << 64) - 1)

    def value(self, scale: float = 1.0) -> float:
        self.state ^= self.state >> 12
        self.state ^= (self.state << 25) & ((1 << 64) - 1)
        self.state ^= self.state >> 27
        raw = (self.state * 2685821657736338717) & ((1 << 64) - 1)
        unit = ((raw >> 40) & 0xFFFFFF) / float(0xFFFFFF)
        return f32(round((unit * 2.0 - 1.0) * scale, 7))

    def values(self, count: int, scale: float = 1.0) -> list[float]:
        return [self.value(scale) for _ in range(count)]


def matmul_rows(left: list[float], right: list[float], rows: int, inner: int, out: int) -> list[float]:
    result = []
    for row in range(rows):
        for target in range(out):
            total = sum(left[row * inner + column] * right[target * inner + column] for column in range(inner))
            result.append(f32(total))
    return result


def rms_norm(values: list[float], weight: list[float], rows: int, width: int, epsilon: float) -> list[float]:
    output = []
    for row in range(rows):
        chunk = values[row * width : (row + 1) * width]
        inverse = 1.0 / math.sqrt(sum(value * value for value in chunk) / width + epsilon)
        output.extend(f32(value * inverse * weight[column]) for column, value in enumerate(chunk))
    return output


def sqrt_softplus(value: float) -> float:
    return f32(math.sqrt(max(value, 0.0) + math.log1p(math.exp(-abs(value)))))


def gate_scores(inputs: list[float], weights: list[float], tokens: int, width: int, experts: int) -> list[float]:
    linear = matmul_rows(inputs, weights, tokens, width, experts)
    return [sqrt_softplus(value) for value in linear]


def route_weights(scores: list[float], indices: list[int], tokens: int, experts: int, top_k: int, scale: float) -> list[float]:
    output = []
    for token in range(tokens):
        selected = [scores[token * experts + indices[token * top_k + slot]] for slot in range(top_k)]
        denominator = sum(selected)
        output.extend(f32(value / denominator * scale) for value in selected)
    return output


def learned_route(inputs: list[float], weights: list[float], bias: list[float], tokens: int, width: int, experts: int, top_k: int, scale: float) -> tuple[list[int], list[float]]:
    scores = gate_scores(inputs, weights, tokens, width, experts)
    indices = []
    for token in range(tokens):
        order = sorted(range(experts), key=lambda expert: (-f32(scores[token * experts + expert] + bias[expert]), expert))
        indices.extend(order[:top_k])
    return indices, route_weights(scores, indices, tokens, experts, top_k, scale)


def expert_swiglu(inputs: list[float], w1: list[float], w2: list[float], w3: list[float], routes: list[float], tokens: int, width: int, intermediate: int, limit: float) -> list[float]:
    gate = matmul_rows(inputs, w1, tokens, width, intermediate)
    up = matmul_rows(inputs, w3, tokens, width, intermediate)
    activated = []
    for token in range(tokens):
        for column in range(intermediate):
            gate_value = min(gate[token * intermediate + column], limit)
            up_value = max(-limit, min(up[token * intermediate + column], limit))
            silu = gate_value / (1.0 + math.exp(-gate_value))
            activated.append(f32(silu * up_value * routes[token]))
    return matmul_rows(activated, w2, tokens, intermediate, width)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return f32(1.0 / (1.0 + math.exp(-value)))
    exponential = math.exp(value)
    return f32(exponential / (1.0 + exponential))


def hc_split(mixes: list[float], scale: list[float], base: list[float], tokens: int, multiplier: int, iterations: int, epsilon: float) -> tuple[list[float], list[float], list[float]]:
    mix_width = (2 + multiplier) * multiplier
    pre: list[float] = []
    post: list[float] = []
    combination: list[float] = []
    for token in range(tokens):
        offset = token * mix_width
        pre.extend(f32(sigmoid(f32(mixes[offset + i] * scale[0] + base[i])) + epsilon) for i in range(multiplier))
        post.extend(f32(2.0 * sigmoid(f32(mixes[offset + multiplier + i] * scale[1] + base[multiplier + i]))) for i in range(multiplier))
        matrix = []
        for row in range(multiplier):
            logits = [
                f32(mixes[offset + 2 * multiplier + row * multiplier + column] * scale[2]
                    + base[2 * multiplier + row * multiplier + column])
                for column in range(multiplier)
            ]
            maximum = max(logits)
            exponentials = [f32(math.exp(value - maximum)) for value in logits]
            denominator = sum(exponentials)
            matrix.append([f32(value / denominator + epsilon) for value in exponentials])

        def normalize_columns() -> None:
            for column in range(multiplier):
                denominator = sum(matrix[row][column] for row in range(multiplier)) + epsilon
                for row in range(multiplier):
                    matrix[row][column] = f32(matrix[row][column] / denominator)

        normalize_columns()
        for _ in range(iterations - 1):
            for row in range(multiplier):
                denominator = sum(matrix[row]) + epsilon
                matrix[row] = [f32(value / denominator) for value in matrix[row]]
            normalize_columns()
        combination.extend(value for row in matrix for value in row)
    return pre, post, combination


def hc_pre(inputs: list[float], weights: list[float], scale: list[float], base: list[float], tokens: int, multiplier: int, width: int, iterations: int, norm_epsilon: float, hc_epsilon: float) -> tuple[list[float], list[float], list[float], list[float]]:
    flat_width = multiplier * width
    mix_width = (2 + multiplier) * multiplier
    mixes = []
    for token in range(tokens):
        row = inputs[token * flat_width : (token + 1) * flat_width]
        inverse = 1.0 / math.sqrt(sum(value * value for value in row) / flat_width + norm_epsilon)
        for target in range(mix_width):
            value = sum(row[column] * weights[target * flat_width + column] for column in range(flat_width))
            mixes.append(f32(value * inverse))
    pre, post, combination = hc_split(mixes, scale, base, tokens, multiplier, iterations, hc_epsilon)
    reduced = []
    for token in range(tokens):
        for column in range(width):
            value = sum(pre[token * multiplier + copy] * inputs[(token * multiplier + copy) * width + column]
                        for copy in range(multiplier))
            reduced.append(f32(value))
    return reduced, pre, post, combination


def hc_post(sublayer: list[float], residual: list[float], post: list[float], combination: list[float], tokens: int, multiplier: int, width: int) -> list[float]:
    output = []
    for token in range(tokens):
        for destination in range(multiplier):
            for column in range(width):
                value = post[token * multiplier + destination] * sublayer[token * width + column]
                # combination is stored [source][destination]. The Sinkhorn softmax runs over the
                # inner index and its last normalization is over sources, so this is the direction
                # that sums to one; the transpose sums a direction nothing normalized and multiplies
                # the residual a little more each layer.
                #
                # This file carried the same error as the C++ it was written to check, which is why
                # the two agreed for a milestone. Two implementations agreeing proves nothing when
                # one was written from the other's reading of the equations.
                value += sum(
                    combination[(token * multiplier + source) * multiplier + destination]
                    * residual[(token * multiplier + source) * width + column]
                    for source in range(multiplier)
                )
                output.append(f32(value))
    return output


def apply_rope(inputs: list[float], positions: list[int], rows: int, heads: int, width: int, theta: float, inverse: bool = False) -> list[float]:
    output = list(inputs)
    for row in range(rows):
        for head in range(heads):
            offset = (row * heads + head) * width
            for pair in range(width // 2):
                frequency = theta ** (-2.0 * pair / width)
                angle = positions[row] * frequency * (-1.0 if inverse else 1.0)
                real, imaginary = inputs[offset + 2 * pair : offset + 2 * pair + 2]
                output[offset + 2 * pair] = f32(real * math.cos(angle) - imaginary * math.sin(angle))
                output[offset + 2 * pair + 1] = f32(imaginary * math.cos(angle) + real * math.sin(angle))
    return output


def sparse_attention(query: list[float], kv: list[float], sink: list[float], indices: list[int], queries: int, kv_count: int, heads: int, width: int, selected: int, scale: float) -> list[float]:
    output = []
    for token in range(queries):
        for head in range(heads):
            scores = []
            for slot in range(selected):
                index = indices[token * selected + slot]
                if index < 0:
                    scores.append(float("-inf"))
                    continue
                dot = sum(query[(token * heads + head) * width + column] * kv[index * width + column]
                          for column in range(width))
                scores.append(dot * scale)
            maximum = max(scores)
            denominator = math.exp(sink[head] - maximum)
            denominator += sum(math.exp(value - maximum) for value in scores if math.isfinite(value))
            for column in range(width):
                numerator = sum(
                    math.exp(scores[slot] - maximum) * kv[indices[token * selected + slot] * width + column]
                    for slot in range(selected)
                    if indices[token * selected + slot] >= 0
                )
                output.append(f32(numerator / denominator))
    return output


def build_fixture() -> dict[str, object]:
    generator = Generator(0xD33F5EE4)

    rms_input = generator.values(8, 1.2)
    rms_weight = generator.values(4, 0.4)
    rms_weight = [f32(value + 1.0) for value in rms_weight]

    route_input = generator.values(8, 0.9)
    gate_weight = generator.values(16, 0.7)
    bias = [0.02, -0.03, 0.05, -0.01]
    learned_indices, learned_weights = learned_route(route_input, gate_weight, bias, 2, 4, 4, 2, 1.5)
    token_ids = [1, 4]
    token_to_experts = [0, 1, 1, 3, 2, 0, 3, 2, 1, 0, 2, 3]
    hash_indices = token_to_experts[2:4] + token_to_experts[8:10]
    scores = gate_scores(route_input, gate_weight, 2, 4, 4)
    hash_weights = route_weights(scores, hash_indices, 2, 4, 2, 1.5)

    expert_input = generator.values(8, 1.1)
    w1 = generator.values(12, 0.8)
    w2 = generator.values(12, 0.8)
    w3 = generator.values(12, 0.8)
    expert_routes = [0.7, 1.2]

    hc_input = generator.values(12, 0.9)
    hc_weight = generator.values(48, 0.5)
    hc_scale = [0.8, -0.6, 1.1]
    hc_base = generator.values(8, 0.2)
    reduced, hc_pre_values, hc_post_values, combination = hc_pre(
        hc_input, hc_weight, hc_scale, hc_base, 2, 2, 3, 4, 1e-6, 1e-6
    )
    sublayer = generator.values(6, 0.7)

    rope_input = generator.values(16, 1.0)
    positions = [3, 7]
    rope_output = apply_rope(rope_input, positions, 2, 2, 4, 10000.0)

    attention_query = generator.values(16, 0.8)
    attention_kv = generator.values(20, 0.8)
    attention_sink = [0.15, -0.2]
    attention_indices = [0, 1, -1, 1, 2, 4, 3, -1]

    return {
        "format": "vastloom.deepseek_v4.operator_fixture.v1",
        "official_revision": REVISION,
        "reference": "Pure-Python translation of pinned model.py/kernel.py equations",
        "absolute_tolerance": 3e-5,
        "rms_norm": {
            "rows": 2,
            "width": 4,
            "epsilon": 1e-6,
            "input": rms_input,
            "weight": rms_weight,
            "expected": rms_norm(rms_input, rms_weight, 2, 4, 1e-6),
        },
        "learned_route": {
            "tokens": 2,
            "width": 4,
            "experts": 4,
            "top_k": 2,
            "route_scale": 1.5,
            "input": route_input,
            "gate_weight": gate_weight,
            "selection_bias": bias,
            "expected_indices": learned_indices,
            "expected_weights": learned_weights,
        },
        "hash_route": {
            "tokens": 2,
            "width": 4,
            "vocabulary": 6,
            "experts": 4,
            "top_k": 2,
            "route_scale": 1.5,
            "input": route_input,
            "gate_weight": gate_weight,
            "token_ids": token_ids,
            "token_to_experts": token_to_experts,
            "expected_indices": hash_indices,
            "expected_weights": hash_weights,
        },
        "expert": {
            "tokens": 2,
            "width": 4,
            "intermediate": 3,
            "swiglu_limit": 0.35,
            "input": expert_input,
            "gate_weight": w1,
            "down_weight": w2,
            "up_weight": w3,
            "routing_weights": expert_routes,
            "expected": expert_swiglu(expert_input, w1, w2, w3, expert_routes, 2, 4, 3, 0.35),
        },
        "hyper_connection": {
            "tokens": 2,
            "multiplier": 2,
            "width": 3,
            "sinkhorn_iterations": 4,
            "norm_epsilon": 1e-6,
            "hc_epsilon": 1e-6,
            "input": hc_input,
            "mixing_weight": hc_weight,
            "scale": hc_scale,
            "base": hc_base,
            "sublayer": sublayer,
            "expected_reduced": reduced,
            "expected_pre": hc_pre_values,
            "expected_post": hc_post_values,
            "expected_combination": combination,
            "expected_output": hc_post(sublayer, hc_input, hc_post_values, combination, 2, 2, 3),
        },
        "rope": {
            "rows": 2,
            "heads": 2,
            "width": 4,
            "theta": 10000.0,
            "positions": positions,
            "input": rope_input,
            "expected": rope_output,
            "expected_inverse": apply_rope(rope_output, positions, 2, 2, 4, 10000.0, True),
        },
        "sparse_attention": {
            "queries": 2,
            "kv_count": 5,
            "heads": 2,
            "width": 4,
            "selected": 4,
            "softmax_scale": 0.5,
            "query": attention_query,
            "key_value": attention_kv,
            "attention_sink": attention_sink,
            "selected_indices": attention_indices,
            "expected": sparse_attention(attention_query, attention_kv, attention_sink, attention_indices, 2, 5, 2, 4, 4, 0.5),
        },
    }


def render() -> str:
    return json.dumps(build_fixture(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "deepseek_v4_operators.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    content = render()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"fixture differs from deterministic reference: {output}")
        print(f"fixture reproducibility: PASS ({output})")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    json.loads(temporary_path.read_text(encoding="utf-8"))
    os.replace(temporary_path, output)
    print(f"wrote {output} ({len(content.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

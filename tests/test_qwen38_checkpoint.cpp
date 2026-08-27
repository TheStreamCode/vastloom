// Opt-in real-checkpoint test for the Qwen3.8-Flash-Next FP8 catalog.
//
// Enabled by configuring CMake with:
//   -DVASTLOOM_QWEN38_MODEL=<path to the imported unsloth/Qwen3.8-Flash-Next-FP8 root>
//
// The test is metadata-only: it parses config.json and the safetensors index
// and validates the full 152089-tensor catalog against the pinned expected
// invariants. No tensor payload is read.

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "vastloom/core/status.hpp"
#include "vastloom/inference/qwen3_8_flash_next.hpp"

namespace {

int failures = 0;

void check(bool condition, const std::string& label) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << label << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_qwen38_checkpoint <checkpoint-root>\n";
        return 64;
    }
    const std::filesystem::path root(argv[1]);

    using vastloom::Result;
    namespace inf = vastloom::inference;

    const auto config_path = root / "config.json";
    check(std::filesystem::exists(config_path), "config.json exists");
    auto cfg = inf::load_qwen38_flash_next_config(config_path);
    check(static_cast<bool>(cfg), "config parses");
    if (!cfg) {
        std::cerr << "  error: " << cfg.error().message << " (" << cfg.error().context << ")\n";
        return 1;
    }

    const auto& c = cfg.value();
    check(c.hidden_size == inf::Qwen38ExpectedInvariants::hidden_size, "hidden_size == 2560");
    check(c.num_hidden_layers == inf::Qwen38ExpectedInvariants::num_hidden_layers, "layers == 48");
    check(c.num_experts == inf::Qwen38ExpectedInvariants::num_experts, "experts == 512");
    check(c.num_experts_per_tok == inf::Qwen38ExpectedInvariants::num_experts_per_tok, "top-k == 10");
    check(c.vocab_size == inf::Qwen38ExpectedInvariants::vocab_size, "vocab == 248320");
    check(c.full_attention_interval == inf::Qwen38ExpectedInvariants::full_attention_interval,
          "full_attention_interval == 4");

    const auto index_path = root / "model.safetensors.index.json";
    check(std::filesystem::exists(index_path), "index.json exists");
    auto inspected = inf::inspect_qwen38_flash_next_safetensors(index_path, c, /*load_header=*/true);
    check(static_cast<bool>(inspected), "index inspects");
    if (!inspected) {
        std::cerr << "  error: " << inspected.error().message << " (" << inspected.error().context << ")\n";
        return 1;
    }

    check(inspected.value().total_tensors == inf::Qwen38ExpectedCatalog::total_tensors,
          "total tensors == 152089");
    check(inspected.value().total_shards == inf::Qwen38ExpectedCatalog::total_shards,
          "total shards == 131");
    check(inspected.value().fp8_block_scaled_experts == inf::Qwen38ExpectedCatalog::fp8_expert_scale_inv,
          "fp8 scale_inv == 75264");
    check(inspected.value().ngram_tensor_count == inf::Qwen38ExpectedCatalog::ngram_tensors,
          "ngram tensors == 129");
    check(inspected.value().has_gated_residual_global, "global gated residual present");
    check(inspected.value().has_gated_residual_per_layer, "per-layer gated residual present");
    check(inspected.value().shard_loaded, "first shard header loaded");

    // Full catalog validation (the same gate the CLI applies).
    auto validated = inf::qwen38_validate_catalog(inspected.value());
    check(static_cast<bool>(validated), "catalog validates against pinned invariants");
    if (!validated) {
        std::cerr << "  error: " << validated.error().message << " (" << validated.error().context << ")\n";
    } else {
        const auto& r = validated.value();
        check(r.layers.size() == inf::Qwen38ExpectedCatalog::language_layers, "48 language layers");
        check(r.vision_tensors == inf::Qwen38ExpectedCatalog::vision_tensors, "vision == 333");
        check(r.mtp_tensors == inf::Qwen38ExpectedCatalog::mtp_tensors, "mtp == 3101");
        check(r.global_tensors == inf::Qwen38ExpectedCatalog::global_tensors, "globals == 5");
        check(!r.shard_header.tensors.empty(), "shard header tensors non-empty");
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "Qwen3.8-Flash-Next real checkpoint catalog: PASS\n";
    return 0;
}

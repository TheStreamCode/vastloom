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
#include "vastloom/inference/qwen3_8_flash_next_reference.hpp"

namespace {

int failures = 0;
double pinned_sum = 0.0;

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

    // --- Q11c: real layer-0 forward over the imported weights ---
    // BOS token: load trunk + embedding row, run the full layer-0 forward,
    // load the top-k routed FP8 experts, and finish. Asserts finiteness,
    // determinism across two runs, and a consistent router decision.
    {
        namespace ref = vastloom::inference;
        const std::uint32_t bos = ref::Qwen38ExpectedInvariants::bos_token_id;
        ref::Qwen38Layer0Telemetry tel;
        auto weights = ref::qwen38_load_layer0_weights(root, bos, &tel);
        check(static_cast<bool>(weights), "layer0 trunk loads");
        if (!weights) {
            std::cerr << "  error: " << weights.error().message
                      << " (" << weights.error().context << ")\n";
            return 1;
        }
        check(weights.value().token_embed.size()
                  == ref::Qwen38ExpectedInvariants::hidden_size,
              "embedding row width");
        check(weights.value().router.size()
                  == static_cast<std::size_t>(ref::Qwen38ExpectedInvariants::num_experts)
                     * ref::Qwen38ExpectedInvariants::hidden_size,
              "router weight count");

        ref::Qwen38Layer0ForwardResult fwd_storage;
        auto run_forward = [&]() -> ref::Qwen38Layer0ForwardResult* {
            auto fwd = ref::qwen38_layer0_forward(
                weights.value(), bos, ref::Qwen38ExpectedInvariants::num_experts,
                ref::Qwen38ExpectedInvariants::num_experts_per_tok);
            if (!fwd) {
                std::cerr << "  error: " << fwd.error().message
                          << " (" << fwd.error().context << ")\n";
                return nullptr;
            }
            fwd_storage = fwd.take_value();
            return &fwd_storage;
        };

        auto* first = run_forward();
        check(first != nullptr, "layer0 forward runs");
        if (first != nullptr) {
            check(first->route.indices.size()
                      == ref::Qwen38ExpectedInvariants::num_experts_per_tok,
                  "router selects exactly top-k experts");
            check(first->mixed_input.size() == ref::Qwen38ExpectedInvariants::hidden_size,
                  "mixed input width");
            check(first->residual.size()
                      == static_cast<std::size_t>(ref::Qwen38ExpectedInvariants::hc_count)
                         * ref::Qwen38ExpectedInvariants::hidden_size,
                  "mlp residual stream width");

            // Load each selected FP8 expert and accumulate the routed sum.
            const std::uint32_t H = ref::Qwen38ExpectedInvariants::hidden_size;
            const std::uint32_t I = ref::Qwen38ExpectedInvariants::moe_intermediate_size;
            std::vector<float> expert_sum(H, 0.0F);
            bool experts_ok = true;
            for (std::size_t i = 0; i < first->route.indices.size(); ++i) {
                auto expert = ref::qwen38_load_layer0_expert(root, first->route.indices[i], &tel);
                if (!expert) {
                    experts_ok = false;
                    std::cerr << "  error: " << expert.error().message
                              << " (" << expert.error().context << ")\n";
                    break;
                }
                auto y = ref::qwen38_expert_mlp(first->mixed_input,
                                                expert.value().gate_proj,
                                                expert.value().up_proj,
                                                expert.value().down_proj, H, I);
                if (!y) {
                    experts_ok = false;
                    std::cerr << "  error: " << y.error().message
                              << " (" << y.error().context << ")\n";
                    break;
                }
                for (std::uint32_t d = 0; d < H; ++d) {
                    expert_sum[d] += y.value()[d] * first->route.weights[i];
                }
            }
            check(experts_ok, "selected experts load and run");
            if (experts_ok) {
                auto finished = ref::qwen38_layer0_finish(*first, expert_sum);
                check(static_cast<bool>(finished), "layer0 finish combines");
                if (finished) {
                    bool finite = true;
                    for (const float v : finished.value()) {
                        if (!std::isfinite(v)) { finite = false; break; }
                    }
                    check(finite, "layer0 output finite");
                    double sum = 0.0;
                    for (const float v : finished.value()) sum += static_cast<double>(v);
                    std::cout << "layer-0 output sum: " << sum << "\n";
                    pinned_sum = sum;

                    // Determinism: a second full run must reproduce the exact sum.
                    auto* second = run_forward();
                    check(second != nullptr, "layer0 forward reruns");
                    if (second != nullptr) {
                        std::vector<float> expert_sum2(H, 0.0F);
                        for (std::size_t i = 0; i < second->route.indices.size(); ++i) {
                            auto expert = ref::qwen38_load_layer0_expert(
                                root, second->route.indices[i], &tel);
                            if (!expert) { experts_ok = false; break; }
                            auto y = ref::qwen38_expert_mlp(second->mixed_input,
                                                            expert.value().gate_proj,
                                                            expert.value().up_proj,
                                                            expert.value().down_proj, H, I);
                            if (!y) { experts_ok = false; break; }
                            for (std::uint32_t d = 0; d < H; ++d) {
                                expert_sum2[d] += y.value()[d] * second->route.weights[i];
                            }
                        }
                        auto finished2 = ref::qwen38_layer0_finish(*second, expert_sum2);
                        check(static_cast<bool>(finished2) && experts_ok,
                              "second forward completes");
                        if (finished2 && experts_ok) {
                            double sum2 = 0.0;
                            for (const float v : finished2.value()) {
                                sum2 += static_cast<double>(v);
                            }
                            check(sum2 == pinned_sum, "layer0 forward deterministic");
                        }
                    }
                }
            }
        }
        check(tel.tensors_read >= 22U, "at least the 22 trunk tensors read");
        check(tel.expert_payload_bytes > 0U, "expert payload was read");
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "Qwen3.8-Flash-Next real checkpoint layer-0 forward: PASS\n";
    return 0;
}

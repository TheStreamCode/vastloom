// Opt-in real-checkpoint test for the Qwen3.8-Flash-Next FP8 catalog.
//
// Enabled by configuring CMake with:
//   -DVASTLOOM_QWEN38_MODEL=<path to the imported unsloth/Qwen3.8-Flash-Next-FP8 root>
//
// The test is metadata-only: it parses config.json and the safetensors index
// and validates the full 152089-tensor catalog against the pinned expected
// invariants. No tensor payload is read.

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "vastloom/core/status.hpp"
#include "vastloom/inference/qwen3_8_flash_next.hpp"
#include "vastloom/inference/qwen3_8_flash_next_reference.hpp"

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

    // --- Q11c/Q11d: real layer-0 forward + multi-token prefill ---
    // One token: trunk -> forward -> routed FP8 experts -> finish, asserting
    // finiteness and bit-identical determinism. Then a 5-token prefill carries
    // the DeltaNet state across tokens through a bounded expert LRU; a full
    // replay from scratch must reproduce the exact output sums.
    {
        namespace ref = vastloom::inference;
        const std::uint32_t bos = ref::Qwen38ExpectedInvariants::bos_token_id;
        const std::uint32_t H = ref::Qwen38ExpectedInvariants::hidden_size;
        const std::uint32_t I = ref::Qwen38ExpectedInvariants::moe_intermediate_size;
        ref::Qwen38Layer0Telemetry tel;
        auto trunk = ref::qwen38_load_layer0_trunk(root, &tel);
        check(static_cast<bool>(trunk), "layer0 trunk loads");
        if (!trunk) {
            std::cerr << "  error: " << trunk.error().message
                      << " (" << trunk.error().context << ")\n";
            return 1;
        }
        check(trunk.value().expert_shards.size()
                  == ref::Qwen38ExpectedInvariants::num_experts,
              "expert shard map covers 512 experts");
        check(trunk.value().router.size()
                  == static_cast<std::size_t>(ref::Qwen38ExpectedInvariants::num_experts) * H,
              "router weight count");

        // Runs one token: caller-owned DeltaNet state and expert LRU.
        auto run_token = [&](ref::Qwen38DeltaNetState& state,
                             ref::Qwen38ExpertLru& lru,
                             std::uint32_t tid,
                             double* out_sum) -> bool {
            auto row = ref::qwen38_load_embed_row(root, tid, &tel);
            if (!row) {
                std::cerr << "  error: " << row.error().message
                          << " (" << row.error().context << ")\n";
                return false;
            }
            auto fwd = ref::qwen38_layer0_forward(
                trunk.value(), row.value(), state,
                ref::Qwen38ExpectedInvariants::num_experts,
                ref::Qwen38ExpectedInvariants::num_experts_per_tok);
            if (!fwd) {
                std::cerr << "  error: " << fwd.error().message
                          << " (" << fwd.error().context << ")\n";
                return false;
            }
            if (fwd.value().route.indices.size()
                    != ref::Qwen38ExpectedInvariants::num_experts_per_tok) {
                return false;
            }
            std::vector<float> expert_sum(H, 0.0F);
            for (std::size_t i = 0; i < fwd.value().route.indices.size(); ++i) {
                const auto eid = fwd.value().route.indices[i];
                static const ref::Qwen38OwnedExpert kEmpty;
                const bool cached = std::find(lru.keys.begin(), lru.keys.end(), eid)
                                    != lru.keys.end();
                const ref::Qwen38OwnedExpert* ptr = nullptr;
                if (cached) {
                    auto acquired = ref::qwen38_expert_cache_acquire(lru, eid, kEmpty);
                    if (!acquired) return false;
                    ptr = acquired.value();
                } else {
                    auto real = ref::qwen38_load_layer0_expert(
                        trunk.value(), root, eid, &tel);
                    if (!real) {
                        std::cerr << "  error: " << real.error().message
                                  << " (" << real.error().context << ")\n";
                        return false;
                    }
                    auto acquired = ref::qwen38_expert_cache_acquire(lru, eid, real.value());
                    if (!acquired) return false;
                    ptr = acquired.value();
                }
                auto y = ref::qwen38_expert_mlp(fwd.value().mixed_input,
                                                ptr->gate_proj, ptr->up_proj,
                                                ptr->down_proj, H, I);
                if (!y) return false;
                for (const float v : y.value()) {
                    if (!std::isfinite(v)) return false;
                }
                for (std::uint32_t d = 0; d < H; ++d) {
                    expert_sum[d] += y.value()[d] * fwd.value().route.weights[i];
                }
            }
            auto finished = ref::qwen38_layer0_finish(fwd.value(), expert_sum);
            if (!finished) return false;
            double sum = 0.0;
            for (const float v : finished.value()) {
                if (!std::isfinite(v)) return false;
                sum += static_cast<double>(v);
            }
            *out_sum = sum;
            return true;
        };

        // Fresh state/cache pair per pass.
        auto fresh = []() {
            std::pair<ref::Qwen38DeltaNetState, ref::Qwen38ExpertLru> p;
            p.second.capacity = 16U;
            return p;
        };

        auto pass1 = fresh();
        double bos_sum = 0.0;
        check(run_token(pass1.first, pass1.second, bos, &bos_sum),
              "layer0 single-token forward runs");
        std::cout << "layer-0 output sum (BOS): " << bos_sum << "\n";

        auto pass2 = fresh();
        double bos_sum2 = 0.0;
        check(run_token(pass2.first, pass2.second, bos, &bos_sum2),
              "layer0 single-token rerun");
        check(bos_sum == bos_sum2, "single-token forward deterministic");

        // Multi-token prefill: the DeltaNet state carries across tokens.
        const std::array<std::uint32_t, 5U> seq{bos, 100U, 200U, 300U, 400U};
        auto prefill = [&](std::vector<double>* sums) -> bool {
            auto s = fresh();
            for (const std::uint32_t tid : seq) {
                double v = 0.0;
                if (!run_token(s.first, s.second, tid, &v)) {
                    return false;
                }
                sums->push_back(v);
            }
            return true;
        };
        std::vector<double> sums;
        check(prefill(&sums), "5-token prefill runs");
        if (!sums.empty()) {
            check(sums[0] == bos_sum, "prefill token 0 matches isolated run");
            bool distinct = false;
            for (std::size_t i = 1; i < sums.size(); ++i) {
                if (sums[i] != sums[0]) { distinct = true; break; }
            }
            check(distinct, "state carry changes later-token outputs");
            std::cout << "prefill sums:";
            for (const double s : sums) std::cout << ' ' << s;
            std::cout << "\n";
            std::vector<double> replay;
            check(prefill(&replay), "prefill replay runs");
            check(replay == sums, "prefill replay deterministic");
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

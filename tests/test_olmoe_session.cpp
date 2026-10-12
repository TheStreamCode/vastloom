// Opt-in session-equivalence test for the OLMoE streamed runtime.
//
// Enabled by configuring CMake with:
//   -DVASTLOOM_OLMOE_FULL_MODEL=<complete pinned OLMoE checkpoint>
//
// A forward through a primed OlmoeSession must produce bit-identical logits to
// the stateless forward of the same token sequence, and must report how many
// leading tokens were served from the cached Key/Value prefix. Divergence,
// overflow and invalid capacities degrade to the stateless path, never to
// wrong output.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "vastloom/core/status.hpp"
#include "vastloom/hardware/hardware_probe.hpp"
#include "vastloom/inference/olmoe_runtime.hpp"
#include "vastloom/safety/resource_governor.hpp"
#include "vastloom/safety/runtime_policy.hpp"

namespace {

int failures = 0;

void check(bool condition, const std::string& label) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << label << "\n";
    }
}

float max_abs_diff(const std::vector<float>& left, const std::vector<float>& right) {
    float worst = 0.0F;
    const std::size_t count = std::min(left.size(), right.size());
    for (std::size_t index = 0U; index < count; ++index) {
        worst = std::max(worst, std::fabs(left[index] - right[index]));
    }
    return worst;
}

std::uint32_t greedy_id(const std::vector<float>& logits) {
    return static_cast<std::uint32_t>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_olmoe_session <olmoe-checkpoint-root>\n";
        return 64;
    }
    const std::filesystem::path root(argv[1]);

    using vastloom::Result;
    namespace inf = vastloom::inference;

    auto hardware = vastloom::hardware::HardwareProbe::collect(root);
    check(static_cast<bool>(hardware), "hardware probe collects");
    if (!hardware) {
        std::cerr << "  error: " << hardware.error().message << "\n";
        return 1;
    }
    auto policy = vastloom::safety::make_safe_runtime_policy(hardware.value());
    check(static_cast<bool>(policy), "safe runtime policy resolves");
    if (!policy) {
        std::cerr << "  error: " << policy.error().message << "\n";
        return 1;
    }
    auto guard = vastloom::safety::ProcessSafetyGuard::apply(policy.value());
    check(static_cast<bool>(guard), "process guard applies");
    if (!guard) {
        std::cerr << "  error: " << guard.error().message << "\n";
        return 1;
    }
    auto governor =
        std::make_unique<vastloom::safety::ResourceGovernor>(policy.value().resources);
    auto model = inf::OlmoeRuntimeModel::load(
        root, *governor, {true, policy.value().cpu_threads});
    check(static_cast<bool>(model), "OLMoE checkpoint loads with verification");
    if (!model) {
        std::cerr << "  error: " << model.error().message << " (" << model.error().context
                  << ")\n";
        return 1;
    }
    const auto& config = model.value().config();

    // Prefix/tail/diverging sequences built from the pinned tokenizer. The tail
    // is appended as encoded pieces so the prefix relationship holds by
    // construction, regardless of how BPE would segment the joined text.
    auto prefix_ids = model.value().tokenizer().encode("Hello");
    auto tail_ids = model.value().tokenizer().encode(" world");
    auto other_ids = model.value().tokenizer().encode(" there");
    check(static_cast<bool>(prefix_ids) && static_cast<bool>(tail_ids) &&
              static_cast<bool>(other_ids),
          "prompt pieces encode");
    if (!prefix_ids || !tail_ids || !other_ids) return 1;
    check(!prefix_ids.value().empty() && !tail_ids.value().empty() &&
              !other_ids.value().empty(),
          "prompt pieces are non-empty");
    check(tail_ids.value().front() != other_ids.value().front(),
          "tails diverge at the first token");
    std::vector<std::uint32_t> full_ids = prefix_ids.value();
    full_ids.insert(full_ids.end(), tail_ids.value().begin(), tail_ids.value().end());
    std::vector<std::uint32_t> diverged_ids = prefix_ids.value();
    diverged_ids.insert(diverged_ids.end(), other_ids.value().begin(),
                        other_ids.value().end());

    const std::size_t capacity =
        std::min<std::size_t>(config.max_position_embeddings, 64U);
    check(full_ids.size() < capacity && diverged_ids.size() < capacity,
          "sequences fit the session capacity");
    if (!(full_ids.size() < capacity && diverged_ids.size() < capacity)) return 1;

    // Invalid capacities are refused before any reservation.
    check(!model.value().create_session(*governor, 0U),
          "zero capacity refused");
    check(!model.value().create_session(
              *governor, static_cast<std::size_t>(config.max_position_embeddings) + 1U),
          "over-context capacity refused");

    const inf::OlmoeForwardOptions options;

    // Stateless reference: the full sequence evaluated without a session.
    auto reference = model.value().forward(full_ids, options, *governor, &guard.value());
    check(static_cast<bool>(reference), "stateless reference forward succeeds");
    if (!reference) {
        std::cerr << "  error: " << reference.error().message << "\n";
        return 1;
    }
    check(reference.value().stats.cached_prefix_tokens == 0U,
          "stateless forward reports zero cached tokens");
    check(reference.value().logits.size() == config.vocab_size, "logits cover vocab");

    // Primed session: the prefix primes the cache, the full sequence reuses it.
    auto created = model.value().create_session(*governor, capacity);
    check(static_cast<bool>(created), "session creates within budget");
    if (!created) {
        std::cerr << "  error: " << created.error().message << "\n";
        return 1;
    }
    inf::OlmoeSession session = created.take_value();
    check(session.ready() && session.capacity() == capacity, "session is ready");
    auto primed =
        model.value().forward(prefix_ids.value(), options, *governor, &guard.value(), &session);
    check(static_cast<bool>(primed), "prefix prime forward succeeds");
    if (!primed) {
        std::cerr << "  error: " << primed.error().message << "\n";
        return 1;
    }
    check(primed.value().stats.cached_prefix_tokens == 0U,
          "cold prime reports zero cached tokens");
    check(session.cached_token_count() == prefix_ids.value().size(),
          "prime caches the prefix length");
    auto reused =
        model.value().forward(full_ids, options, *governor, &guard.value(), &session);
    check(static_cast<bool>(reused), "prefix-reuse forward succeeds");
    if (!reused) {
        std::cerr << "  error: " << reused.error().message << "\n";
        return 1;
    }
    check(reused.value().stats.cached_prefix_tokens == prefix_ids.value().size(),
          "reuse reports the cached prefix length");
    check(reused.value().logits.size() == reference.value().logits.size(),
          "reuse logits match reference size");
    const float reuse_diff = max_abs_diff(reused.value().logits, reference.value().logits);
    check(reuse_diff == 0.0F, "reuse logits are bit-identical to stateless");
    if (reuse_diff != 0.0F) {
        std::cerr << "  max abs diff: " << reuse_diff << "\n";
    }
    check(greedy_id(reused.value().logits) == greedy_id(reference.value().logits),
          "reuse greedy token matches stateless");

    // Divergence: a sequence that forks inside the cached prefix degrades to
    // the stateless path with identical output.
    auto diverged_reference =
        model.value().forward(diverged_ids, options, *governor, &guard.value());
    check(static_cast<bool>(diverged_reference), "diverged stateless reference succeeds");
    if (!diverged_reference) {
        std::cerr << "  error: " << diverged_reference.error().message << "\n";
        return 1;
    }
    auto diverged =
        model.value().forward(diverged_ids, options, *governor, &guard.value(), &session);
    check(static_cast<bool>(diverged), "diverged session forward succeeds");
    if (!diverged) {
        std::cerr << "  error: " << diverged.error().message << "\n";
        return 1;
    }
    check(diverged.value().stats.cached_prefix_tokens == 0U,
          "divergence reports zero cached tokens");
    const float diverged_diff =
        max_abs_diff(diverged.value().logits, diverged_reference.value().logits);
    check(diverged_diff == 0.0F, "diverged output matches stateless");
    if (diverged_diff != 0.0F) {
        std::cerr << "  max abs diff: " << diverged_diff << "\n";
    }

    // Overflow: a request longer than the session capacity degrades the same way.
    auto small = model.value().create_session(*governor, prefix_ids.value().size());
    check(static_cast<bool>(small), "minimal session creates");
    if (!small) {
        std::cerr << "  error: " << small.error().message << "\n";
        return 1;
    }
    inf::OlmoeSession tiny = small.take_value();
    auto tiny_prime = model.value().forward(prefix_ids.value(), options, *governor,
                                            &guard.value(), &tiny);
    check(static_cast<bool>(tiny_prime), "minimal session prime succeeds");
    auto overflowed =
        model.value().forward(full_ids, options, *governor, &guard.value(), &tiny);
    check(static_cast<bool>(overflowed), "overflowing forward succeeds");
    if (!overflowed) {
        std::cerr << "  error: " << overflowed.error().message << "\n";
        return 1;
    }
    check(overflowed.value().stats.cached_prefix_tokens == 0U,
          "overflow reports zero cached tokens");
    const float overflow_diff =
        max_abs_diff(overflowed.value().logits, reference.value().logits);
    check(overflow_diff == 0.0F, "overflowing output matches stateless");
    if (overflow_diff != 0.0F) {
        std::cerr << "  max abs diff: " << overflow_diff << "\n";
    }

    if (failures == 0) {
        std::cout << "OLMoE session equivalence: ALL PASSED\n";
    } else {
        std::cerr << "OLMoE session equivalence: " << failures << " FAILURES\n";
    }
    return failures == 0 ? 0 : 1;
}

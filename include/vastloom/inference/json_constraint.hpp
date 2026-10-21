#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "vastloom/core/status.hpp"
#include "vastloom/inference/token_constraint.hpp"

namespace vastloom::inference {

// Hard bounds for the JSON-object constraint. Every limit is enforced while the
// constraint is being fed, so a malformed or adversarial model cannot grow the
// state without bound.
struct JsonConstraintLimits {
    // Nested object/array depth, root object included.
    std::uint32_t max_depth{8U};
    // Bytes of one string body between its quotes; escape sequences count as
    // written. This bounds the automaton state, not the decoded value.
    std::size_t max_string_bytes{1024U};
    // Bytes of one JSON number literal.
    std::size_t max_number_bytes{64U};
    // Upper bound on the token-byte table built by initialize().
    std::size_t max_token_bytes_total{16U * 1024U * 1024U};
    // Upper bound on the vocabulary size the constraint will mask.
    std::size_t max_vocab_entries{262'144U};
};

// Grammar-constrained decoding to exactly one JSON object as the root value.
// The accepted language is the RFC 8259 JSON text grammar restricted to:
//   - root object only (`{ ... }`), no leading/trailing non-whitespace bytes;
//   - the configured depth, string-length and number-length ceilings;
//   - UTF-8 validated inside strings.
// JSON Schema compilation is a separate future constraint; this class is the
// first TokenConstraint implementation and deliberately has no schema input.
class JsonObjectConstraint final : public TokenConstraint {
public:
    [[nodiscard]] static Result<JsonObjectConstraint> create(
        const JsonConstraintLimits& limits = {});
    ~JsonObjectConstraint() override;

    JsonObjectConstraint(JsonObjectConstraint&&) noexcept;
    JsonObjectConstraint& operator=(JsonObjectConstraint&&) noexcept;
    JsonObjectConstraint(const JsonObjectConstraint&) = delete;
    JsonObjectConstraint& operator=(const JsonObjectConstraint&) = delete;

    [[nodiscard]] Result<void> initialize(const Gpt2Tokenizer& tokenizer) override;
    [[nodiscard]] Result<std::vector<std::uint8_t>> allowed_mask() override;
    [[nodiscard]] Result<void> accept_token(std::uint32_t token_id) override;
    [[nodiscard]] bool complete() const noexcept override;
    void reset() override;

private:
    struct Impl;
    explicit JsonObjectConstraint(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace vastloom::inference

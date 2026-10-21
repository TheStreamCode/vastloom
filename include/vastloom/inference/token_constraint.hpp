#pragma once

#include <cstdint>
#include <vector>

#include "vastloom/core/status.hpp"

namespace vastloom::inference {

class Gpt2Tokenizer;

// A per-request decoder constraint. The generation loop asks for a token mask
// before sampling and reports the sampled token back afterwards. Implementations
// own all mutable state; the caller owns the object and guarantees it outlives
// the generation it constrains.
//
// A constraint never changes the target model's arithmetic: it only removes
// tokens from the sampling distribution. The mask is checked against the same
// vocabulary the model produces logits for, so a masked token can be sampled
// only if the constraint is internally inconsistent.
class TokenConstraint {
public:
    virtual ~TokenConstraint() = default;

    // Binds the constraint to `tokenizer`'s vocabulary. Called once before
    // generation; the tokenizer must outlive the constraint.
    [[nodiscard]] virtual Result<void> initialize(const Gpt2Tokenizer& tokenizer) = 0;

    // One byte per vocabulary entry: 1 when the token may be emitted at the
    // current state, 0 otherwise. An empty vector means "not initialized".
    [[nodiscard]] virtual Result<std::vector<std::uint8_t>> allowed_mask() = 0;

    // Consumes one sampled token id. Called only for tokens the mask allowed.
    [[nodiscard]] virtual Result<void> accept_token(std::uint32_t token_id) = 0;

    // True when the constrained document is complete (EOS is legal from here).
    [[nodiscard]] virtual bool complete() const noexcept = 0;

    // Returns to the state the constraint had after initialize().
    virtual void reset() = 0;
};

}  // namespace vastloom::inference

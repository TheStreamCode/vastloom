#include "vastloom/inference/json_constraint.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "vastloom/inference/gpt2_tokenizer.hpp"

namespace vastloom::inference {

namespace {

// Contract ceilings: JsonConstraintLimits may only narrow these, never widen them.
constexpr std::uint32_t kMaxDepthCeiling = 32U;
constexpr std::size_t kMaxStringBytesCeiling = 1024U * 1024U;      // 1 MiB
constexpr std::size_t kMaxNumberBytesCeiling = 1024U;              // 1 KiB
constexpr std::size_t kMaxVocabEntriesCeiling = 1024U * 1024U;     // 1 Mi entries

[[nodiscard]] constexpr bool is_json_whitespace(std::uint8_t byte) noexcept {
    return byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D;
}

[[nodiscard]] constexpr bool is_digit(std::uint8_t byte) noexcept {
    return byte >= '0' && byte <= '9';
}

[[nodiscard]] constexpr bool is_one_through_nine(std::uint8_t byte) noexcept {
    return byte >= '1' && byte <= '9';
}

[[nodiscard]] constexpr bool is_hex_digit(std::uint8_t byte) noexcept {
    return is_digit(byte) || (byte >= 'a' && byte <= 'f') || (byte >= 'A' && byte <= 'F');
}

// The grammar is replayed one byte at a time because a vocabulary token may end
// in the middle of any construct (a string escape, a number, "true", ...): only
// the next byte can legalise the continuation.
enum class Mode : std::uint8_t {
    dead,
    expect_root,  // optional whitespace, then the root '{'
    after_root,   // root object closed: whitespace only
    object_first,           // first member: '"' or '}'
    object_key,             // a key is required (reached after a comma)
    object_after_key,       // ':' is expected
    object_value,           // a value follows the key's colon
    object_comma_or_close,  // ',' or '}'
    array_first,            // first element: value or ']'
    array_value,            // a value is required (reached after a comma)
    array_comma_or_close,   // ',' or ']'
    string_body,
    string_escape,
    string_unicode,
    number_minus,
    number_zero,
    number_int,
    number_frac_start,
    number_frac,
    number_exp_start,
    number_exp_sign,
    number_exp,
    literal,
};

enum class Frame : std::uint8_t { object, array };

// Everything allowed_mask() needs to replay one token. Deliberately a trivially
// copyable POD with a fixed frame stack: a mask simulation copies it once per
// vocabulary entry and must never allocate.
struct SimState {
    std::array<Frame, kMaxDepthCeiling> frames{};
    Mode mode{Mode::expect_root};
    std::uint8_t depth{0};
    std::uint8_t literal_id{0};
    std::uint8_t literal_index{0};
    std::uint8_t unicode_digits{0};
    std::uint8_t utf8_remaining{0};
    std::uint8_t utf8_min{0};
    std::uint8_t utf8_max{0};
    bool key_string{false};
    std::size_t string_bytes{0};
    std::size_t number_bytes{0};
};

static_assert(std::is_trivially_copyable_v<SimState>,
              "the per-token simulation state must stay a trivially copyable POD");

[[nodiscard]] bool kill(SimState& state) noexcept {
    state.mode = Mode::dead;
    return false;
}

constexpr std::array<std::uint8_t, 4> kTrueLiteral{'t', 'r', 'u', 'e'};
constexpr std::array<std::uint8_t, 5> kFalseLiteral{'f', 'a', 'l', 's', 'e'};
constexpr std::array<std::uint8_t, 4> kNullLiteral{'n', 'u', 'l', 'l'};

}  // namespace

struct JsonObjectConstraint::Impl {
    JsonConstraintLimits limits{};
    std::vector<std::string> token_bytes;
    std::vector<std::uint8_t> special;
    std::size_t eos_index{0};
    bool initialized{false};
    SimState state{};

    // Feeds one decoded byte into `current`; false means the byte left the
    // accepted language and the state is dead for good.
    [[nodiscard]] bool step(SimState& current, std::uint8_t byte) const noexcept {
        switch (current.mode) {
        case Mode::dead:
            return false;
        case Mode::expect_root:
            if (is_json_whitespace(byte)) return true;
            if (byte == '{') return start_container(current, Frame::object);
            return kill(current);
        case Mode::after_root:
            if (is_json_whitespace(byte)) return true;
            return kill(current);
        case Mode::object_first:
            if (is_json_whitespace(byte)) return true;
            if (byte == '"') {
                start_string(current, true);
                return true;
            }
            if (byte == '}') {
                leave_container(current);
                return true;
            }
            return kill(current);
        case Mode::object_key:
            if (is_json_whitespace(byte)) return true;
            if (byte == '"') {
                start_string(current, true);
                return true;
            }
            return kill(current);
        case Mode::object_after_key:
            if (is_json_whitespace(byte)) return true;
            if (byte == ':') {
                current.mode = Mode::object_value;
                return true;
            }
            return kill(current);
        case Mode::object_value:
            if (is_json_whitespace(byte)) return true;
            return start_value(current, byte);
        case Mode::object_comma_or_close:
            if (is_json_whitespace(byte)) return true;
            if (byte == ',') {
                current.mode = Mode::object_key;
                return true;
            }
            if (byte == '}') {
                leave_container(current);
                return true;
            }
            return kill(current);
        case Mode::array_first:
            if (is_json_whitespace(byte)) return true;
            if (byte == ']') {
                leave_container(current);
                return true;
            }
            return start_value(current, byte);
        case Mode::array_value:
            if (is_json_whitespace(byte)) return true;
            return start_value(current, byte);
        case Mode::array_comma_or_close:
            if (is_json_whitespace(byte)) return true;
            if (byte == ',') {
                current.mode = Mode::array_value;
                return true;
            }
            if (byte == ']') {
                leave_container(current);
                return true;
            }
            return kill(current);
        case Mode::string_body:
            return step_string_body(current, byte);
        case Mode::string_escape:
            return step_string_escape(current, byte);
        case Mode::string_unicode:
            return step_string_unicode(current, byte);
        case Mode::number_minus:
            if (byte == '0') {
                current.mode = Mode::number_zero;
                return true;
            }
            if (is_one_through_nine(byte)) {
                current.mode = Mode::number_int;
                return true;
            }
            return kill(current);
        case Mode::number_zero:
            // A leading zero may only be followed by a fraction or an exponent;
            // any other byte terminates an already complete integer part.
            if (byte == '.') return extend_number(current, Mode::number_frac_start);
            if (byte == 'e' || byte == 'E') return extend_number(current, Mode::number_exp_start);
            finish_value(current);
            return step(current, byte);
        case Mode::number_int:
            if (is_digit(byte)) return extend_number(current, Mode::number_int);
            if (byte == '.') return extend_number(current, Mode::number_frac_start);
            if (byte == 'e' || byte == 'E') return extend_number(current, Mode::number_exp_start);
            finish_value(current);
            return step(current, byte);
        case Mode::number_frac_start:
            if (is_digit(byte)) return extend_number(current, Mode::number_frac);
            return kill(current);
        case Mode::number_frac:
            if (is_digit(byte)) return extend_number(current, Mode::number_frac);
            if (byte == 'e' || byte == 'E') return extend_number(current, Mode::number_exp_start);
            finish_value(current);
            return step(current, byte);
        case Mode::number_exp_start:
            if (byte == '+' || byte == '-') return extend_number(current, Mode::number_exp_sign);
            if (is_digit(byte)) return extend_number(current, Mode::number_exp);
            return kill(current);
        case Mode::number_exp_sign:
            if (is_digit(byte)) return extend_number(current, Mode::number_exp);
            return kill(current);
        case Mode::number_exp:
            if (is_digit(byte)) return extend_number(current, Mode::number_exp);
            finish_value(current);
            return step(current, byte);
        case Mode::literal:
            return extend_literal(current, byte);
        }
        // Only an out-of-range Mode value (never produced by this DFA) can reach here.
        return kill(current);
    }

    [[nodiscard]] bool start_value(SimState& current, std::uint8_t byte) const noexcept {
        if (byte == '{') return start_container(current, Frame::object);
        if (byte == '[') return start_container(current, Frame::array);
        if (byte == '"') {
            start_string(current, false);
            return true;
        }
        if (byte == '-') {
            current.mode = Mode::number_minus;
            current.number_bytes = 1U;
            return true;
        }
        if (byte == '0') {
            current.mode = Mode::number_zero;
            current.number_bytes = 1U;
            return true;
        }
        if (is_one_through_nine(byte)) {
            current.mode = Mode::number_int;
            current.number_bytes = 1U;
            return true;
        }
        if (byte == 't' || byte == 'f' || byte == 'n') {
            current.mode = Mode::literal;
            current.literal_id = static_cast<std::uint8_t>(byte == 't' ? 0 : (byte == 'f' ? 1 : 2));
            current.literal_index = 1U;
            return true;
        }
        return kill(current);
    }

    [[nodiscard]] bool start_container(SimState& current, Frame frame) const noexcept {
        // The root object is depth 1, so the new frame is legal while the stack
        // is still shallower than max_depth.
        if (static_cast<std::uint32_t>(current.depth) >= limits.max_depth) return kill(current);
        current.frames[static_cast<std::size_t>(current.depth)] = frame;
        ++current.depth;
        current.mode = frame == Frame::object ? Mode::object_first : Mode::array_first;
        return true;
    }

    void leave_container(SimState& current) const noexcept {
        // Callers only leave a container the DFA is currently inside of.
        --current.depth;
        if (current.depth == 0) {
            // Brackets inside the root object only close the root object, so
            // reaching depth zero means the whole document value is complete.
            current.mode = Mode::after_root;
            return;
        }
        finish_value(current);
    }

    void finish_value(SimState& current) const noexcept {
        // A value only completes inside a container, so depth is at least one.
        current.mode = current.frames[static_cast<std::size_t>(current.depth - 1U)] == Frame::object
                           ? Mode::object_comma_or_close
                           : Mode::array_comma_or_close;
    }

    void start_string(SimState& current, bool key) const noexcept {
        current.mode = Mode::string_body;
        current.key_string = key;
        current.string_bytes = 0U;
        current.utf8_remaining = 0;
        current.unicode_digits = 0;
    }

    [[nodiscard]] bool step_string_body(SimState& current, std::uint8_t byte) const noexcept {
        if (byte == '"') {
            if (current.utf8_remaining != 0) return kill(current);
            if (current.key_string) {
                current.mode = Mode::object_after_key;
            } else {
                finish_value(current);
            }
            return true;
        }
        if (byte == '\\') {
            if (current.utf8_remaining != 0) return kill(current);
            if (!consume_string_byte(current)) return false;
            current.mode = Mode::string_escape;
            return true;
        }
        if (!consume_string_byte(current)) return false;
        return consume_utf8(current, byte);
    }

    [[nodiscard]] bool step_string_escape(SimState& current, std::uint8_t byte) const noexcept {
        if (!consume_string_byte(current)) return false;
        switch (byte) {
        case '"':
        case '\\':
        case '/':
        case 'b':
        case 'f':
        case 'n':
        case 'r':
        case 't':
            current.mode = Mode::string_body;
            return true;
        case 'u':
            current.mode = Mode::string_unicode;
            current.unicode_digits = 0;
            return true;
        default:
            return kill(current);
        }
    }

    [[nodiscard]] bool step_string_unicode(SimState& current, std::uint8_t byte) const noexcept {
        if (!consume_string_byte(current)) return false;
        if (!is_hex_digit(byte)) return kill(current);
        ++current.unicode_digits;
        if (current.unicode_digits == 4) {
            current.unicode_digits = 0;
            current.mode = Mode::string_body;
        }
        return true;
    }

    [[nodiscard]] bool consume_string_byte(SimState& current) const noexcept {
        ++current.string_bytes;
        if (current.string_bytes > limits.max_string_bytes) return kill(current);
        return true;
    }

    [[nodiscard]] bool consume_number_byte(SimState& current) const noexcept {
        ++current.number_bytes;
        if (current.number_bytes > limits.max_number_bytes) return kill(current);
        return true;
    }

    [[nodiscard]] bool extend_number(SimState& current, Mode next_mode) const noexcept {
        if (!consume_number_byte(current)) return false;
        current.mode = next_mode;
        return true;
    }

    // Incremental UTF-8 validation for string content: the grammar accepts any
    // byte that is not a control, so well-formedness needs its own check.
    [[nodiscard]] bool consume_utf8(SimState& current, std::uint8_t byte) const noexcept {
        if (current.utf8_remaining != 0) {
            if (byte < current.utf8_min || byte > current.utf8_max) return kill(current);
            --current.utf8_remaining;
            current.utf8_min = static_cast<std::uint8_t>(0x80);
            current.utf8_max = static_cast<std::uint8_t>(0xBF);
            return true;
        }
        if (byte < 0x80) {
            // Raw C0 controls are not legal inside a JSON string.
            if (byte < 0x20) return kill(current);
            return true;
        }
        if (byte >= 0xC2 && byte <= 0xDF) {
            current.utf8_remaining = 1;
            current.utf8_min = static_cast<std::uint8_t>(0x80);
            current.utf8_max = static_cast<std::uint8_t>(0xBF);
            return true;
        }
        if (byte == 0xE0) {
            current.utf8_remaining = 2;
            current.utf8_min = static_cast<std::uint8_t>(0xA0);
            current.utf8_max = static_cast<std::uint8_t>(0xBF);
            return true;
        }
        if ((byte >= 0xE1 && byte <= 0xEC) || (byte >= 0xEE && byte <= 0xEF)) {
            current.utf8_remaining = 2;
            current.utf8_min = static_cast<std::uint8_t>(0x80);
            current.utf8_max = static_cast<std::uint8_t>(0xBF);
            return true;
        }
        if (byte == 0xED) {
            // Excludes the UTF-16 surrogate range, which UTF-8 must not encode.
            current.utf8_remaining = 2;
            current.utf8_min = static_cast<std::uint8_t>(0x80);
            current.utf8_max = static_cast<std::uint8_t>(0x9F);
            return true;
        }
        if (byte == 0xF0) {
            current.utf8_remaining = 3;
            current.utf8_min = static_cast<std::uint8_t>(0x90);
            current.utf8_max = static_cast<std::uint8_t>(0xBF);
            return true;
        }
        if (byte >= 0xF1 && byte <= 0xF3) {
            current.utf8_remaining = 3;
            current.utf8_min = static_cast<std::uint8_t>(0x80);
            current.utf8_max = static_cast<std::uint8_t>(0xBF);
            return true;
        }
        if (byte == 0xF4) {
            current.utf8_remaining = 3;
            current.utf8_min = static_cast<std::uint8_t>(0x80);
            current.utf8_max = static_cast<std::uint8_t>(0x8F);
            return true;
        }
        return kill(current);
    }

    [[nodiscard]] bool extend_literal(SimState& current, std::uint8_t byte) const noexcept {
        std::uint8_t expected = 0;
        std::size_t length = 0;
        switch (current.literal_id) {
        case 0:
            expected = kTrueLiteral[static_cast<std::size_t>(current.literal_index)];
            length = kTrueLiteral.size();
            break;
        case 1:
            expected = kFalseLiteral[static_cast<std::size_t>(current.literal_index)];
            length = kFalseLiteral.size();
            break;
        case 2:
            expected = kNullLiteral[static_cast<std::size_t>(current.literal_index)];
            length = kNullLiteral.size();
            break;
        default:
            return kill(current);
        }
        if (byte != expected) return kill(current);
        ++current.literal_index;
        if (static_cast<std::size_t>(current.literal_index) == length) finish_value(current);
        return true;
    }
};

JsonObjectConstraint::JsonObjectConstraint(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
JsonObjectConstraint::~JsonObjectConstraint() = default;
JsonObjectConstraint::JsonObjectConstraint(JsonObjectConstraint&&) noexcept = default;
JsonObjectConstraint& JsonObjectConstraint::operator=(JsonObjectConstraint&&) noexcept = default;

Result<JsonObjectConstraint> JsonObjectConstraint::create(const JsonConstraintLimits& limits) {
    try {
        if (limits.max_depth == 0U || limits.max_depth > kMaxDepthCeiling) {
            return make_error(ErrorCode::invalid_argument,
                              "JSON constraint max_depth is outside the supported range", "1..32");
        }
        if (limits.max_string_bytes == 0U || limits.max_string_bytes > kMaxStringBytesCeiling) {
            return make_error(ErrorCode::invalid_argument,
                              "JSON constraint max_string_bytes is outside the supported range",
                              "1..1048576");
        }
        if (limits.max_number_bytes == 0U || limits.max_number_bytes > kMaxNumberBytesCeiling) {
            return make_error(ErrorCode::invalid_argument,
                              "JSON constraint max_number_bytes is outside the supported range",
                              "1..1024");
        }
        if (limits.max_token_bytes_total == 0U) {
            return make_error(ErrorCode::invalid_argument,
                              "JSON constraint max_token_bytes_total must be at least 1");
        }
        if (limits.max_vocab_entries == 0U || limits.max_vocab_entries > kMaxVocabEntriesCeiling) {
            return make_error(ErrorCode::invalid_argument,
                              "JSON constraint max_vocab_entries is outside the supported range",
                              "1..1048576");
        }
        auto impl = std::make_unique<Impl>();
        impl->limits = limits;
        return JsonObjectConstraint(std::move(impl));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "out of memory");
    } catch (const std::exception&) {
        return make_error(ErrorCode::internal_error, "internal error");
    }
}

Result<void> JsonObjectConstraint::initialize(const Gpt2Tokenizer& tokenizer) {
    try {
        if (!impl_) return make_error(ErrorCode::internal_error, "JSON constraint was moved from");
        // initialize() is idempotent: drop any earlier binding before rebuilding it.
        impl_->initialized = false;
        impl_->token_bytes.clear();
        impl_->special.clear();
        impl_->eos_index = 0U;
        impl_->state = SimState{};

        const std::size_t vocab = tokenizer.vocab_size();
        if (vocab == 0U) {
            return make_error(ErrorCode::invalid_format, "Tokenizer vocabulary is empty");
        }
        if (vocab > impl_->limits.max_vocab_entries) {
            return make_error(ErrorCode::unsupported,
                              "Tokenizer vocabulary exceeds the configured JSON constraint limit",
                              std::to_string(vocab) + " > " +
                                  std::to_string(impl_->limits.max_vocab_entries));
        }
        const std::size_t eos = static_cast<std::size_t>(tokenizer.eos_token_id());
        if (eos >= vocab) {
            return make_error(ErrorCode::integrity_error,
                              "Tokenizer EOS token id is outside the vocabulary");
        }

        std::vector<std::string> bytes;
        std::vector<std::uint8_t> special;
        bytes.reserve(vocab);
        special.resize(vocab, static_cast<std::uint8_t>(0));
        std::size_t total_bytes = 0U;
        for (std::size_t index = 0; index < vocab; ++index) {
            const auto token_id = static_cast<std::uint32_t>(index);
            if (tokenizer.is_special_token(token_id)) {
                // Special tokens are masked except EOS; they contribute no bytes.
                special[index] = static_cast<std::uint8_t>(1);
                bytes.emplace_back();
                continue;
            }
            auto piece = tokenizer.token_bytes(token_id);
            if (!piece) {
                return make_error(ErrorCode::integrity_error,
                                  "Tokenizer token bytes could not be reconstructed",
                                  std::to_string(index));
            }
            if (piece.value().size() > impl_->limits.max_token_bytes_total - total_bytes) {
                return make_error(ErrorCode::resource_exhausted,
                                  "Tokenizer token bytes exceed the configured JSON constraint limit");
            }
            total_bytes += piece.value().size();
            bytes.push_back(piece.take_value());
        }
        impl_->token_bytes = std::move(bytes);
        impl_->special = std::move(special);
        impl_->eos_index = eos;
        impl_->state = SimState{};
        impl_->initialized = true;
        return {};
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "out of memory");
    } catch (const std::exception&) {
        return make_error(ErrorCode::internal_error, "internal error");
    }
}

Result<std::vector<std::uint8_t>> JsonObjectConstraint::allowed_mask() {
    std::vector<std::uint8_t> mask;
    if (!impl_ || !impl_->initialized) return mask;
    try {
        const std::size_t vocab = impl_->token_bytes.size();
        mask.assign(vocab, static_cast<std::uint8_t>(0));
        const bool document_complete = impl_->state.mode == Mode::after_root;
        for (std::size_t index = 0; index < vocab; ++index) {
            if (index == impl_->eos_index) {
                // EOS closes the document; it is legal only once the root object is done.
                if (document_complete) mask[index] = static_cast<std::uint8_t>(1);
                continue;
            }
            if (impl_->special[index] != 0) continue;
            const std::string& bytes = impl_->token_bytes[index];
            if (bytes.empty()) continue;
            SimState probe = impl_->state;
            bool alive = true;
            for (const char raw : bytes) {
                const auto byte = static_cast<std::uint8_t>(static_cast<unsigned char>(raw));
                if (!impl_->step(probe, byte)) {
                    alive = false;
                    break;
                }
            }
            if (alive) mask[index] = static_cast<std::uint8_t>(1);
        }
        return mask;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "out of memory");
    } catch (const std::exception&) {
        return make_error(ErrorCode::internal_error, "internal error");
    }
}

Result<void> JsonObjectConstraint::accept_token(std::uint32_t token_id) {
    try {
        if (!impl_ || !impl_->initialized) {
            return make_error(ErrorCode::internal_error, "JSON constraint is not initialized");
        }
        const std::size_t index = static_cast<std::size_t>(token_id);
        if (index >= impl_->token_bytes.size()) {
            return make_error(ErrorCode::invalid_argument,
                              "Token id is outside the tokenizer vocabulary");
        }
        if (index == impl_->eos_index) {
            if (impl_->state.mode != Mode::after_root) {
                return make_error(ErrorCode::internal_error,
                                  "EOS is only legal once the root JSON object is complete");
            }
            return {};
        }
        if (impl_->special[index] != 0) {
            return make_error(ErrorCode::internal_error,
                              "A special token was accepted by the JSON constraint");
        }
        const std::string& bytes = impl_->token_bytes[index];
        if (bytes.empty()) {
            return make_error(ErrorCode::internal_error,
                              "An empty token was accepted by the JSON constraint");
        }
        SimState probe = impl_->state;
        for (const char raw : bytes) {
            const auto byte = static_cast<std::uint8_t>(static_cast<unsigned char>(raw));
            if (!impl_->step(probe, byte)) {
                return make_error(ErrorCode::internal_error,
                                  "Token does not fit the current JSON grammar state");
            }
        }
        impl_->state = probe;
        return {};
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "out of memory");
    } catch (const std::exception&) {
        return make_error(ErrorCode::internal_error, "internal error");
    }
}

bool JsonObjectConstraint::complete() const noexcept {
    return impl_ && impl_->initialized && impl_->state.mode == Mode::after_root;
}

void JsonObjectConstraint::reset() {
    if (impl_) impl_->state = SimState{};
}

}  // namespace vastloom::inference

#include "stargate/inference/gpt2_tokenizer.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <set>

#include "stargate/core/bytes.hpp"
#include "stargate/util/json.hpp"

namespace stargate::inference {
namespace {

constexpr std::size_t kMaximumVocabEntries = 1'000'000;
constexpr std::size_t kMaximumMergeEntries = 2'000'000;
constexpr std::size_t kMaximumInputBytes = 16U * MiB;
constexpr std::size_t kMaximumMessages = 1024;

struct Codepoint {
    std::string bytes;
    std::uint32_t value{0};
};

Result<std::vector<Codepoint>> split_utf8(std::string_view input) {
    std::vector<Codepoint> output;
    output.reserve(input.size());
    for (std::size_t position = 0; position < input.size();) {
        const auto first = static_cast<unsigned char>(input[position]);
        std::size_t length = 0;
        std::uint32_t value = 0;
        if (first <= 0x7fU) {
            length = 1;
            value = first;
        } else if (first >= 0xc2U && first <= 0xdfU) {
            length = 2;
            value = first & 0x1fU;
        } else if (first >= 0xe0U && first <= 0xefU) {
            length = 3;
            value = first & 0x0fU;
        } else if (first >= 0xf0U && first <= 0xf4U) {
            length = 4;
            value = first & 0x07U;
        } else {
            return make_error(ErrorCode::invalid_argument, "Tokenizer input contains invalid UTF-8", std::to_string(position));
        }
        if (input.size() - position < length) {
            return make_error(ErrorCode::invalid_argument, "Tokenizer input contains truncated UTF-8", std::to_string(position));
        }
        for (std::size_t index = 1; index < length; ++index) {
            const auto continuation = static_cast<unsigned char>(input[position + index]);
            if ((continuation & 0xc0U) != 0x80U) {
                return make_error(ErrorCode::invalid_argument, "Tokenizer input contains invalid UTF-8 continuation", std::to_string(position));
            }
            value = (value << 6U) | (continuation & 0x3fU);
        }
        if ((length == 2U && value < 0x80U) || (length == 3U && value < 0x800U) ||
            (length == 4U && value < 0x10000U) || value > 0x10ffffU ||
            (value >= 0xd800U && value <= 0xdfffU)) {
            return make_error(ErrorCode::invalid_argument, "Tokenizer input contains non-canonical UTF-8", std::to_string(position));
        }
        output.push_back(Codepoint{std::string(input.substr(position, length)), value});
        position += length;
    }
    return output;
}

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7fU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
        output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
        output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
}

bool is_ascii_digit(const Codepoint& codepoint) noexcept {
    return codepoint.value >= '0' && codepoint.value <= '9';
}

bool is_whitespace(const Codepoint& codepoint) noexcept {
    const auto value = codepoint.value;
    return value == 0x20U || (value >= 0x09U && value <= 0x0dU) || value == 0x85U ||
           value == 0xa0U || value == 0x1680U || (value >= 0x2000U && value <= 0x200aU) ||
           value == 0x2028U || value == 0x2029U || value == 0x202fU || value == 0x205fU || value == 0x3000U;
}

bool is_letter(const Codepoint& codepoint) noexcept {
    const auto value = codepoint.value;
    if ((value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z')) return true;
    if ((value >= 0x00c0U && value <= 0x02afU) || (value >= 0x0370U && value <= 0x052fU) ||
        (value >= 0x1e00U && value <= 0x1effU)) return value != 0x00d7U && value != 0x00f7U;
    return value >= 0x4e00U && value <= 0x9fffU;
}

bool contraction_at(const std::vector<Codepoint>& codepoints, std::size_t position, std::size_t* length) {
    if (codepoints[position].value != '\'') return false;
    constexpr std::array<std::string_view, 7> suffixes{"re", "ve", "ll", "s", "t", "m", "d"};
    for (const auto suffix : suffixes) {
        if (position + 1U + suffix.size() > codepoints.size()) continue;
        bool matches = true;
        for (std::size_t index = 0; index < suffix.size(); ++index) {
            if (codepoints[position + 1U + index].bytes.size() != 1U ||
                codepoints[position + 1U + index].bytes[0] != suffix[index]) {
                matches = false;
                break;
            }
        }
        if (matches) {
            *length = suffix.size() + 1U;
            return true;
        }
    }
    return false;
}

std::string merge_key(std::string_view left, std::string_view right) {
    std::string result;
    result.reserve(left.size() + right.size() + 1U);
    result.append(left);
    result.push_back('\0');
    result.append(right);
    return result;
}

}  // namespace

Result<Gpt2Tokenizer> Gpt2Tokenizer::load(const std::filesystem::path& model_directory) {
    const auto vocab_path = model_directory / "vocab.json";
    const auto merges_path = model_directory / "merges.txt";
    const auto config_path = model_directory / "tokenizer_config.json";
    util::JsonLimits vocab_limits;
    vocab_limits.max_input_bytes = 16U * MiB;
    vocab_limits.max_nodes = 2U * kMaximumVocabEntries;
    vocab_limits.max_object_members = kMaximumVocabEntries;
    auto vocab_json = util::load_json_file(vocab_path, vocab_limits);
    if (!vocab_json) return vocab_json.error();
    auto vocab = vocab_json.value().as_object();
    if (!vocab || vocab.value()->empty() || vocab.value()->size() > kMaximumVocabEntries) {
        return make_error(ErrorCode::invalid_format, "Tokenizer vocabulary is invalid", vocab_path.string());
    }

    Gpt2Tokenizer result;
    std::uint64_t maximum_id = 0;
    std::set<std::uint64_t> ids;
    for (const auto& [token, id_value] : *vocab.value()) {
        auto id = id_value.as_u64();
        if (!id || id.value() >= kMaximumVocabEntries || !ids.insert(id.value()).second) {
            return make_error(ErrorCode::invalid_format, "Tokenizer vocabulary id is invalid or duplicated", token);
        }
        maximum_id = std::max(maximum_id, id.value());
    }
    if (maximum_id + 1U != vocab.value()->size()) {
        return make_error(ErrorCode::invalid_format, "Tokenizer vocabulary ids are not contiguous", vocab_path.string());
    }
    result.id_to_token_.resize(static_cast<std::size_t>(maximum_id + 1U));
    result.special_ids_.resize(result.id_to_token_.size(), false);
    result.token_to_id_.reserve(vocab.value()->size());
    for (const auto& [token, id_value] : *vocab.value()) {
        const auto id = static_cast<std::uint32_t>(id_value.as_u64().value());
        result.token_to_id_.emplace(token, id);
        result.id_to_token_[id] = token;
    }

    std::error_code file_error;
    const auto merges_bytes = std::filesystem::file_size(merges_path, file_error);
    if (file_error || merges_bytes > 64U * MiB) {
        return make_error(ErrorCode::resource_exhausted, "Tokenizer merges file is missing or too large", merges_path.string());
    }
    std::ifstream merges(merges_path, std::ios::binary);
    if (!merges) return make_error(ErrorCode::io_error, "Unable to open tokenizer merges", merges_path.string());
    std::string line;
    std::size_t rank = 0;
    while (std::getline(merges, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.starts_with('#')) continue;
        const auto separator = line.find(' ');
        if (separator == std::string::npos || separator == 0U || separator + 1U >= line.size() ||
            line.find(' ', separator + 1U) != std::string::npos || rank >= kMaximumMergeEntries) {
            return make_error(ErrorCode::invalid_format, "Invalid tokenizer merge entry", line);
        }
        if (!result.merge_ranks_.emplace(merge_key(std::string_view(line).substr(0, separator),
                                                   std::string_view(line).substr(separator + 1U)),
                                         rank++).second) {
            return make_error(ErrorCode::invalid_format, "Duplicate tokenizer merge entry", line);
        }
    }

    util::JsonLimits config_limits;
    config_limits.max_input_bytes = 1U * MiB;
    config_limits.max_nodes = 4096;
    auto config_json = util::load_json_file(config_path, config_limits);
    if (!config_json) return config_json.error();
    const auto* added_decoder = config_json.value().find("added_tokens_decoder");
    if (added_decoder == nullptr) return make_error(ErrorCode::invalid_format, "Tokenizer has no added_tokens_decoder", config_path.string());
    auto added = added_decoder->as_object();
    if (!added || added.value()->size() > 4096U) return make_error(ErrorCode::invalid_format, "Tokenizer added-token map is invalid", config_path.string());
    for (const auto& [id_text, definition] : *added.value()) {
        std::uint32_t id = 0;
        const auto converted = std::from_chars(id_text.data(), id_text.data() + id_text.size(), id);
        const auto* content_value = definition.find("content");
        const auto* special_value = definition.find("special");
        if (converted.ec != std::errc{} || converted.ptr != id_text.data() + id_text.size() ||
            id >= result.id_to_token_.size() || content_value == nullptr || special_value == nullptr) {
            return make_error(ErrorCode::invalid_format, "Tokenizer added-token entry is invalid", id_text);
        }
        auto content = content_value->as_string();
        auto special = special_value->as_bool();
        if (!content || !special || result.id_to_token_[id] != content.value()) {
            return make_error(ErrorCode::integrity_error, "Tokenizer added token does not match vocabulary", id_text);
        }
        if (special.value()) {
            result.special_tokens_.push_back(SpecialToken{std::string(content.value()), id});
            result.special_ids_[id] = true;
        }
    }
    std::sort(result.special_tokens_.begin(), result.special_tokens_.end(), [](const auto& left, const auto& right) {
        return left.content.size() > right.content.size();
    });

    std::vector<std::uint32_t> byte_values;
    byte_values.reserve(256);
    for (std::uint32_t value = 33U; value <= 126U; ++value) byte_values.push_back(value);
    for (std::uint32_t value = 161U; value <= 172U; ++value) byte_values.push_back(value);
    for (std::uint32_t value = 174U; value <= 255U; ++value) byte_values.push_back(value);
    std::vector<std::uint32_t> unicode_values = byte_values;
    std::uint32_t extra = 0;
    for (std::uint32_t value = 0; value <= 255U; ++value) {
        if (std::find(byte_values.begin(), byte_values.end(), value) == byte_values.end()) {
            byte_values.push_back(value);
            unicode_values.push_back(256U + extra++);
        }
    }
    for (std::size_t index = 0; index < byte_values.size(); ++index) {
        std::string encoded;
        append_utf8(encoded, unicode_values[index]);
        result.byte_encoder_[byte_values[index]] = encoded;
        result.byte_decoder_.emplace(std::move(encoded), static_cast<std::uint8_t>(byte_values[index]));
    }
    return result;
}

Result<std::vector<std::uint32_t>> Gpt2Tokenizer::encode(std::string_view text) const {
    if (text.size() > kMaximumInputBytes) return make_error(ErrorCode::invalid_argument, "Tokenizer input exceeds 16 MiB");
    std::vector<std::uint32_t> output;
    std::size_t ordinary_start = 0;
    std::size_t position = 0;
    while (position < text.size()) {
        const SpecialToken* matched = nullptr;
        for (const auto& special : special_tokens_) {
            if (text.substr(position, special.content.size()) == special.content) {
                matched = &special;
                break;
            }
        }
        if (matched == nullptr) {
            ++position;
            continue;
        }
        if (position > ordinary_start) {
            auto encoded = encode_ordinary(text.substr(ordinary_start, position - ordinary_start));
            if (!encoded) return encoded.error();
            output.insert(output.end(), encoded.value().begin(), encoded.value().end());
        }
        output.push_back(matched->id);
        position += matched->content.size();
        ordinary_start = position;
    }
    if (ordinary_start < text.size()) {
        auto encoded = encode_ordinary(text.substr(ordinary_start));
        if (!encoded) return encoded.error();
        output.insert(output.end(), encoded.value().begin(), encoded.value().end());
    }
    return output;
}

Result<std::vector<std::uint32_t>> Gpt2Tokenizer::encode_ordinary(std::string_view text) const {
    std::vector<std::uint32_t> output;
    auto pieces = pretokenize(text);
    if (!pieces) return pieces.error();
    for (const auto& piece : pieces.value()) {
        auto encoded = bpe(piece);
        if (!encoded) return encoded.error();
        output.insert(output.end(), encoded.value().begin(), encoded.value().end());
    }
    return output;
}

Result<std::vector<std::string>> Gpt2Tokenizer::pretokenize(std::string_view text) const {
    auto decoded = split_utf8(text);
    if (!decoded) return decoded.error();
    std::vector<std::vector<Codepoint>> digit_pieces;
    std::vector<Codepoint> current;
    for (const auto& codepoint : decoded.value()) {
        if (is_ascii_digit(codepoint)) {
            if (!current.empty()) {
                digit_pieces.push_back(std::move(current));
                current.clear();
            }
            digit_pieces.push_back({codepoint});
        } else {
            current.push_back(codepoint);
        }
    }
    if (!current.empty()) digit_pieces.push_back(std::move(current));

    std::vector<std::string> output;
    for (const auto& codepoints : digit_pieces) {
        for (std::size_t position = 0; position < codepoints.size();) {
            const std::size_t start = position;
            std::size_t contraction_length = 0;
            if (contraction_at(codepoints, position, &contraction_length)) {
                position += contraction_length;
            } else {
                std::size_t content = position;
                if (codepoints[content].value == 0x20U && content + 1U < codepoints.size()) ++content;
                if (is_letter(codepoints[content])) {
                    position = content + 1U;
                    while (position < codepoints.size() && is_letter(codepoints[position])) ++position;
                } else if (is_ascii_digit(codepoints[content])) {
                    position = content + 1U;
                    while (position < codepoints.size() && is_ascii_digit(codepoints[position])) ++position;
                } else if (!is_whitespace(codepoints[content])) {
                    position = content + 1U;
                    while (position < codepoints.size() && !is_whitespace(codepoints[position]) &&
                           !is_letter(codepoints[position]) && !is_ascii_digit(codepoints[position])) ++position;
                } else {
                    position = start + 1U;
                    while (position < codepoints.size() && is_whitespace(codepoints[position])) ++position;
                    if (position < codepoints.size() && position - start > 1U) --position;
                }
            }
            std::string piece;
            for (std::size_t index = start; index < position; ++index) piece += codepoints[index].bytes;
            output.push_back(std::move(piece));
        }
    }
    return output;
}

Result<std::vector<std::uint32_t>> Gpt2Tokenizer::bpe(std::string_view piece) const {
    std::string byte_encoded;
    for (const char raw_byte : piece) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        byte_encoded += byte_encoder_[byte];
    }
    auto split = split_utf8(byte_encoded);
    if (!split) return split.error();
    std::vector<std::string> symbols;
    symbols.reserve(split.value().size());
    for (auto& codepoint : split.value()) symbols.push_back(std::move(codepoint.bytes));
    while (symbols.size() >= 2U) {
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        std::string best_left;
        std::string best_right;
        for (std::size_t index = 0; index + 1U < symbols.size(); ++index) {
            const auto iterator = merge_ranks_.find(merge_key(symbols[index], symbols[index + 1U]));
            if (iterator != merge_ranks_.end() && iterator->second < best_rank) {
                best_rank = iterator->second;
                best_left = symbols[index];
                best_right = symbols[index + 1U];
            }
        }
        if (best_rank == std::numeric_limits<std::size_t>::max()) break;
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t index = 0; index < symbols.size();) {
            if (index + 1U < symbols.size() && symbols[index] == best_left && symbols[index + 1U] == best_right) {
                merged.push_back(symbols[index] + symbols[index + 1U]);
                index += 2U;
            } else {
                merged.push_back(std::move(symbols[index++]));
            }
        }
        symbols = std::move(merged);
    }
    std::vector<std::uint32_t> output;
    output.reserve(symbols.size());
    for (const auto& symbol : symbols) {
        const auto iterator = token_to_id_.find(symbol);
        if (iterator == token_to_id_.end()) return make_error(ErrorCode::integrity_error, "BPE symbol is missing from tokenizer vocabulary", symbol);
        output.push_back(iterator->second);
    }
    return output;
}

Result<std::string> Gpt2Tokenizer::decode(const std::vector<std::uint32_t>& token_ids,
                                          bool skip_special_tokens) const {
    std::string output;
    for (const auto id : token_ids) {
        if (id >= id_to_token_.size()) return make_error(ErrorCode::invalid_argument, "Token id exceeds tokenizer vocabulary", std::to_string(id));
        if (special_ids_[id]) {
            if (!skip_special_tokens) output += id_to_token_[id];
            continue;
        }
        auto symbols = split_utf8(id_to_token_[id]);
        if (!symbols) return make_error(ErrorCode::integrity_error, "Tokenizer vocabulary contains invalid UTF-8", std::to_string(id));
        for (const auto& symbol : symbols.value()) {
            const auto iterator = byte_decoder_.find(symbol.bytes);
            if (iterator == byte_decoder_.end()) return make_error(ErrorCode::integrity_error, "Tokenizer byte symbol has no reverse mapping", symbol.bytes);
            output.push_back(static_cast<char>(iterator->second));
        }
    }
    return output;
}

Result<std::string> Gpt2Tokenizer::apply_smol_chat_template(const std::vector<ChatMessage>& messages,
                                                            bool add_generation_prompt) const {
    if (messages.empty() || messages.size() > kMaximumMessages) {
        return make_error(ErrorCode::invalid_argument, "Chat requires between 1 and 1024 messages");
    }
    constexpr std::string_view default_system =
        "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
    std::string output;
    const auto append_message = [&output](std::string_view role, std::string_view content) {
        output += "<|im_start|>";
        output += role;
        output.push_back('\n');
        output += content;
        output += "<|im_end|>\n";
    };
    if (messages.front().role != "system") append_message("system", default_system);
    for (const auto& message : messages) {
        if ((message.role != "system" && message.role != "user" && message.role != "assistant") ||
            message.content.size() > kMaximumInputBytes) {
            return make_error(ErrorCode::invalid_argument, "Invalid chat role or oversized message", message.role);
        }
        append_message(message.role, message.content);
    }
    if (add_generation_prompt) output += "<|im_start|>assistant\n";
    if (output.size() > kMaximumInputBytes) return make_error(ErrorCode::invalid_argument, "Formatted chat exceeds 16 MiB");
    return output;
}

}  // namespace stargate::inference

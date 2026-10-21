#include "vastloom/inference/gpt2_tokenizer.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <tuple>

#include "vastloom/core/bytes.hpp"
#include "vastloom/util/json.hpp"

namespace vastloom::inference {
namespace {

constexpr std::size_t kMaximumVocabEntries = 1'000'000;
constexpr std::size_t kMaximumMergeEntries = 2'000'000;
constexpr std::size_t kMaximumInputBytes = 16U * MiB;
constexpr std::size_t kMaximumMessages = 1024;

struct Codepoint {
    std::string bytes;
    std::uint32_t value{0};
};

#include "unicode_categories.inc"

template <std::size_t Size>
bool in_unicode_ranges(std::uint32_t value,
                       const std::array<UnicodeRange, Size>& ranges) noexcept {
    std::size_t left = 0U;
    std::size_t right = ranges.size();
    while (left < right) {
        const std::size_t middle = left + (right - left) / 2U;
        if (value < ranges[middle].first) {
            right = middle;
        } else if (value > ranges[middle].last) {
            left = middle + 1U;
        } else {
            return true;
        }
    }
    return false;
}

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

bool is_number(const Codepoint& codepoint) noexcept {
    return in_unicode_ranges(codepoint.value, kUnicodeNRanges);
}

bool is_whitespace(const Codepoint& codepoint) noexcept {
    const auto value = codepoint.value;
    return (value >= 0x09U && value <= 0x0dU) || value == 0x85U ||
           in_unicode_ranges(value, kUnicodeZRanges);
}

bool is_letter(const Codepoint& codepoint) noexcept {
    return in_unicode_ranges(codepoint.value, kUnicodeLRanges);
}

bool is_mark(const Codepoint& codepoint) noexcept {
    return in_unicode_ranges(codepoint.value, kUnicodeMRanges);
}

bool is_punctuation(const Codepoint& codepoint) noexcept {
    return in_unicode_ranges(codepoint.value, kUnicodePRanges);
}

bool is_symbol(const Codepoint& codepoint) noexcept {
    return in_unicode_ranges(codepoint.value, kUnicodeSRanges);
}

bool is_ascii_letter(const Codepoint& codepoint) noexcept {
    return (codepoint.value >= 'A' && codepoint.value <= 'Z') ||
           (codepoint.value >= 'a' && codepoint.value <= 'z');
}

bool is_ascii_pattern_punctuation(const Codepoint& codepoint) noexcept {
    if (codepoint.value > 0x7fU) return false;
    constexpr std::string_view punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    return punctuation.find(static_cast<char>(codepoint.value)) != std::string_view::npos;
}

bool is_newline(const Codepoint& codepoint) noexcept {
    return codepoint.value == '\r' || codepoint.value == '\n';
}

bool is_deepseek_cjk(const Codepoint& codepoint) noexcept {
    const auto value = codepoint.value;
    return (value >= 0x4e00U && value <= 0x9fa5U) ||
           (value >= 0x3040U && value <= 0x309fU) ||
           (value >= 0x30a0U && value <= 0x30ffU);
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

bool json_string_is(const util::JsonValue& object,
                    std::string_view key,
                    std::string_view expected) {
    const auto* value = object.find(key);
    if (value == nullptr) return false;
    auto text = value->as_string();
    return text && text.value() == expected;
}

bool json_bool_is(const util::JsonValue& object, std::string_view key, bool expected) {
    const auto* value = object.find(key);
    if (value == nullptr) return false;
    auto parsed = value->as_bool();
    return parsed && parsed.value() == expected;
}

Result<void> validate_deepseek_v4_tokenizer_layout(const util::JsonValue& root,
                                                   const std::filesystem::path& source) {
    const auto* normalizer = root.find("normalizer");
    const auto* pretokenizer = root.find("pre_tokenizer");
    const auto* decoder = root.find("decoder");
    const auto* postprocessor = root.find("post_processor");
    if (normalizer == nullptr || pretokenizer == nullptr || decoder == nullptr || postprocessor == nullptr ||
        !json_string_is(*normalizer, "type", "Sequence") ||
        !json_string_is(*pretokenizer, "type", "Sequence") ||
        !json_string_is(*decoder, "type", "ByteLevel") ||
        !json_string_is(*postprocessor, "type", "ByteLevel")) {
        return make_error(ErrorCode::unsupported, "Tokenizer is not the DeepSeek V4 byte-BPE layout", source.string());
    }
    const auto* normalizers_value = normalizer->find("normalizers");
    const auto* pretokenizers_value = pretokenizer->find("pretokenizers");
    auto normalizers = normalizers_value == nullptr
                           ? Result<const util::JsonValue::Array*>(make_error(ErrorCode::invalid_format, "missing normalizers"))
                           : normalizers_value->as_array();
    auto pretokenizers = pretokenizers_value == nullptr
                             ? Result<const util::JsonValue::Array*>(make_error(ErrorCode::invalid_format, "missing pretokenizers"))
                             : pretokenizers_value->as_array();
    if (!normalizers || !normalizers.value()->empty() || !pretokenizers || pretokenizers.value()->size() != 4U) {
        return make_error(ErrorCode::unsupported, "DeepSeek V4 tokenizer sequence differs from the pinned layout", source.string());
    }
    constexpr std::array<std::string_view, 3> patterns{
        R"(\p{N}{1,3})",
        R"([一-龥぀-ゟ゠-ヿ]+)",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
    };
    for (std::size_t index = 0U; index < patterns.size(); ++index) {
        const auto& split = (*pretokenizers.value())[index];
        const auto* pattern = split.find("pattern");
        if (!json_string_is(split, "type", "Split") || !json_string_is(split, "behavior", "Isolated") ||
            !json_bool_is(split, "invert", false) || pattern == nullptr ||
            !json_string_is(*pattern, "Regex", patterns[index])) {
            return make_error(ErrorCode::unsupported, "DeepSeek V4 tokenizer split regex differs from the pinned layout", std::to_string(index));
        }
    }
    const auto& byte_level = (*pretokenizers.value())[3U];
    if (!json_string_is(byte_level, "type", "ByteLevel") ||
        !json_bool_is(byte_level, "add_prefix_space", false) ||
        !json_bool_is(byte_level, "trim_offsets", true) ||
        !json_bool_is(byte_level, "use_regex", false)) {
        return make_error(ErrorCode::unsupported, "DeepSeek V4 ByteLevel pre-tokenizer differs from the pinned layout", source.string());
    }
    return {};
}

Result<std::string_view> tokenizer_config_token_content(const util::JsonValue& value,
                                                        std::string_view field) {
    if (value.is_string()) return value.as_string();
    if (value.is_object()) {
        const auto* content = value.find("content");
        if (content != nullptr) return content->as_string();
    }
    return make_error(ErrorCode::invalid_format, "Tokenizer special-token field has no string content", std::string(field));
}

void append_codepoint_piece(const std::vector<Codepoint>& codepoints,
                            std::size_t begin,
                            std::size_t end,
                            std::vector<std::string>& output) {
    std::string piece;
    for (std::size_t index = begin; index < end; ++index) piece += codepoints[index].bytes;
    if (!piece.empty()) output.push_back(std::move(piece));
}

std::optional<std::size_t> deepseek_general_match_end(const std::vector<Codepoint>& codepoints,
                                                      std::size_t position,
                                                      std::size_t limit) {
    const auto advance_letters = [&](std::size_t cursor) {
        while (cursor < limit && (is_letter(codepoints[cursor]) || is_mark(codepoints[cursor]))) ++cursor;
        return cursor;
    };
    if (is_ascii_pattern_punctuation(codepoints[position]) && position + 1U < limit &&
        is_ascii_letter(codepoints[position + 1U])) {
        std::size_t cursor = position + 2U;
        while (cursor < limit && is_ascii_letter(codepoints[cursor])) ++cursor;
        return cursor;
    }
    if (is_letter(codepoints[position]) || is_mark(codepoints[position])) {
        return advance_letters(position + 1U);
    }
    const bool optional_letter_prefix = !is_newline(codepoints[position]) &&
                                        !is_letter(codepoints[position]) &&
                                        !is_punctuation(codepoints[position]) &&
                                        !is_symbol(codepoints[position]);
    if (optional_letter_prefix && position + 1U < limit &&
        (is_letter(codepoints[position + 1U]) || is_mark(codepoints[position + 1U]))) {
        return advance_letters(position + 2U);
    }
    std::size_t punctuation_start = position;
    if (codepoints[position].value == 0x20U && position + 1U < limit &&
        (is_punctuation(codepoints[position + 1U]) || is_symbol(codepoints[position + 1U]))) {
        punctuation_start = position + 1U;
    }
    if (is_punctuation(codepoints[punctuation_start]) || is_symbol(codepoints[punctuation_start])) {
        std::size_t cursor = punctuation_start + 1U;
        while (cursor < limit && (is_punctuation(codepoints[cursor]) || is_symbol(codepoints[cursor]))) ++cursor;
        while (cursor < limit && is_newline(codepoints[cursor])) ++cursor;
        return cursor;
    }
    if (is_whitespace(codepoints[position])) {
        std::size_t cursor = position;
        std::optional<std::size_t> last_newline;
        while (cursor < limit && is_whitespace(codepoints[cursor])) {
            if (is_newline(codepoints[cursor])) last_newline = cursor;
            ++cursor;
        }
        if (last_newline) return *last_newline + 1U;
        if (cursor == limit) return cursor;
        if (cursor - position > 1U) return cursor - 1U;
        return cursor;
    }
    return std::nullopt;
}

std::vector<std::string> deepseek_pretokenize(const std::vector<Codepoint>& codepoints) {
    std::vector<std::string> output;
    for (std::size_t position = 0U; position < codepoints.size();) {
        if (is_number(codepoints[position])) {
            std::size_t end = position + 1U;
            while (end < codepoints.size() && end - position < 3U && is_number(codepoints[end])) ++end;
            append_codepoint_piece(codepoints, position, end, output);
            position = end;
            continue;
        }
        if (is_deepseek_cjk(codepoints[position])) {
            std::size_t end = position + 1U;
            while (end < codepoints.size() && is_deepseek_cjk(codepoints[end])) ++end;
            append_codepoint_piece(codepoints, position, end, output);
            position = end;
            continue;
        }
        std::size_t segment_end = position + 1U;
        while (segment_end < codepoints.size() && !is_number(codepoints[segment_end]) &&
               !is_deepseek_cjk(codepoints[segment_end])) {
            ++segment_end;
        }
        while (position < segment_end) {
            auto matched_end = deepseek_general_match_end(codepoints, position, segment_end);
            if (matched_end) {
                append_codepoint_piece(codepoints, position, *matched_end, output);
                position = *matched_end;
                continue;
            }
            std::size_t unmatched_end = position + 1U;
            while (unmatched_end < segment_end &&
                   !deepseek_general_match_end(codepoints, unmatched_end, segment_end)) {
                ++unmatched_end;
            }
            append_codepoint_piece(codepoints, position, unmatched_end, output);
            position = unmatched_end;
        }
    }
    return output;
}

}  // namespace

Result<Gpt2Tokenizer> Gpt2Tokenizer::load(const std::filesystem::path& model_directory) {
    const auto vocab_path = model_directory / "vocab.json";
    const auto merges_path = model_directory / "merges.txt";
    const auto tokenizer_path = model_directory / "tokenizer.json";
    const auto config_path = model_directory / "tokenizer_config.json";
    util::JsonLimits vocab_limits;
    vocab_limits.max_input_bytes = 16U * MiB;
    vocab_limits.max_nodes = 3U * kMaximumVocabEntries;
    vocab_limits.max_object_members = kMaximumVocabEntries;
    vocab_limits.max_array_elements = kMaximumMergeEntries;

    Gpt2Tokenizer result;
    std::vector<std::string> inline_merges;
    struct PendingAddedToken {
        std::uint32_t id{0};
        std::string content;
        bool special{false};
    };
    std::vector<PendingAddedToken> pending_added_tokens;
    const auto populate_vocab = [&](const util::JsonValue::Object& vocab, const std::filesystem::path& source) -> Result<void> {
        if (vocab.empty() || vocab.size() > kMaximumVocabEntries) {
            return make_error(ErrorCode::invalid_format, "Tokenizer vocabulary is invalid", source.string());
        }
        std::uint64_t maximum_id = 0;
        std::set<std::uint64_t> ids;
        for (const auto& [token, id_value] : vocab) {
            auto id = id_value.as_u64();
            if (!id || id.value() >= kMaximumVocabEntries || !ids.insert(id.value()).second) {
                return make_error(ErrorCode::invalid_format, "Tokenizer vocabulary id is invalid or duplicated", token);
            }
            maximum_id = std::max(maximum_id, id.value());
        }
        if (maximum_id + 1U != vocab.size()) {
            return make_error(ErrorCode::invalid_format, "Tokenizer vocabulary ids are not contiguous", source.string());
        }
        result.id_to_token_.resize(static_cast<std::size_t>(maximum_id + 1U));
        result.added_ids_.resize(result.id_to_token_.size(), false);
        result.special_ids_.resize(result.id_to_token_.size(), false);
        result.token_to_id_.reserve(vocab.size() + 64U);
        for (const auto& [token, id_value] : vocab) {
            const auto id = static_cast<std::uint32_t>(id_value.as_u64().value());
            result.token_to_id_.emplace(token, id);
            result.id_to_token_[id] = token;
        }
        return {};
    };

    std::error_code file_error;
    if (std::filesystem::is_regular_file(vocab_path, file_error)) {
        auto vocab_json = util::load_json_file(vocab_path, vocab_limits);
        if (!vocab_json) return vocab_json.error();
        auto vocab = vocab_json.value().as_object();
        if (!vocab) return make_error(ErrorCode::invalid_format, "Tokenizer vocabulary root is not an object", vocab_path.string());
        auto populated = populate_vocab(*vocab.value(), vocab_path);
        if (!populated) return populated.error();
        result.split_digits_individually_ = true;
    } else {
        file_error.clear();
        auto tokenizer_json = util::load_json_file(tokenizer_path, vocab_limits);
        if (!tokenizer_json) return tokenizer_json.error();
        const auto* model_value = tokenizer_json.value().find("model");
        const auto* pretokenizer_value = tokenizer_json.value().find("pre_tokenizer");
        const auto* normalizer_value = tokenizer_json.value().find("normalizer");
        if (model_value == nullptr || pretokenizer_value == nullptr || normalizer_value == nullptr) {
            return make_error(ErrorCode::invalid_format, "Tokenizer JSON is missing model/pre-tokenizer/normalizer", tokenizer_path.string());
        }
        const auto* type_value = model_value->find("type");
        const auto* vocab_value = model_value->find("vocab");
        const auto* merges_value = model_value->find("merges");
        const auto* pre_type_value = pretokenizer_value->find("type");
        const auto* normalizer_type_value = normalizer_value->find("type");
        if (type_value == nullptr || vocab_value == nullptr || merges_value == nullptr ||
            pre_type_value == nullptr || normalizer_type_value == nullptr) {
            return make_error(ErrorCode::invalid_format, "Tokenizer JSON BPE fields are missing", tokenizer_path.string());
        }
        auto type = type_value->as_string();
        auto vocab = vocab_value->as_object();
        auto merges = merges_value->as_array();
        auto pre_type = pre_type_value->as_string();
        auto normalizer_type = normalizer_type_value->as_string();
        if (!type || type.value() != "BPE" || !vocab || !merges || merges.value()->size() > kMaximumMergeEntries ||
            !pre_type || !normalizer_type) {
            return make_error(ErrorCode::unsupported, "Tokenizer JSON is not a supported byte BPE", tokenizer_path.string());
        }
        if (pre_type.value() == "ByteLevel" && normalizer_type.value() == "NFC") {
            result.pretokenizer_mode_ = PretokenizerMode::gpt2;
        } else if (pre_type.value() == "Sequence" && normalizer_type.value() == "Sequence") {
            auto layout = validate_deepseek_v4_tokenizer_layout(tokenizer_json.value(), tokenizer_path);
            if (!layout) return layout.error();
            result.pretokenizer_mode_ = PretokenizerMode::deepseek_v4;
        } else {
            return make_error(ErrorCode::unsupported, "Tokenizer byte-BPE pre-tokenizer is unsupported", tokenizer_path.string());
        }
        auto populated = populate_vocab(*vocab.value(), tokenizer_path);
        if (!populated) return populated.error();
        inline_merges.reserve(merges.value()->size());
        for (const auto& merge_value : *merges.value()) {
            auto merge = merge_value.as_string();
            if (!merge) return make_error(ErrorCode::invalid_format, "Tokenizer merge is not a string", tokenizer_path.string());
            inline_merges.emplace_back(merge.value());
        }
        result.split_digits_individually_ = false;
        if (result.pretokenizer_mode_ == PretokenizerMode::deepseek_v4) {
            const auto* added_value = tokenizer_json.value().find("added_tokens");
            auto added = added_value == nullptr
                             ? Result<const util::JsonValue::Array*>(make_error(ErrorCode::invalid_format, "missing added_tokens"))
                             : added_value->as_array();
            if (!added || added.value()->size() > 4096U) {
                return make_error(ErrorCode::invalid_format, "DeepSeek tokenizer added-token list is invalid", tokenizer_path.string());
            }
            pending_added_tokens.reserve(added.value()->size());
            for (const auto& definition : *added.value()) {
                const auto* id_value = definition.find("id");
                const auto* content_value = definition.find("content");
                const auto* special_value = definition.find("special");
                const auto* single_word_value = definition.find("single_word");
                const auto* lstrip_value = definition.find("lstrip");
                const auto* rstrip_value = definition.find("rstrip");
                if (id_value == nullptr || content_value == nullptr || special_value == nullptr ||
                    single_word_value == nullptr || lstrip_value == nullptr || rstrip_value == nullptr) {
                    return make_error(ErrorCode::invalid_format, "DeepSeek tokenizer added-token fields are missing", tokenizer_path.string());
                }
                auto id = id_value->as_u64();
                auto content = content_value->as_string();
                auto special = special_value->as_bool();
                auto single_word = single_word_value->as_bool();
                auto left_strip = lstrip_value->as_bool();
                auto right_strip = rstrip_value->as_bool();
                if (!id || id.value() >= kMaximumVocabEntries || !content || content.value().empty() || !special ||
                    !single_word || single_word.value() || !left_strip || left_strip.value() ||
                    !right_strip || right_strip.value()) {
                    return make_error(ErrorCode::unsupported, "DeepSeek tokenizer uses an unsupported added-token rule", tokenizer_path.string());
                }
                pending_added_tokens.push_back(PendingAddedToken{
                    static_cast<std::uint32_t>(id.value()), std::string(content.value()), special.value()});
            }
            std::sort(pending_added_tokens.begin(), pending_added_tokens.end(),
                      [](const auto& left, const auto& right) { return left.id < right.id; });
        }
    }

    std::size_t rank = 0;
    const auto add_merge = [&](std::string line) -> Result<void> {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.starts_with('#')) return {};
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
        return {};
    };
    if (!inline_merges.empty()) {
        for (const auto& merge : inline_merges) {
            auto added = add_merge(merge);
            if (!added) return added.error();
        }
    } else {
        const auto merges_bytes = std::filesystem::file_size(merges_path, file_error);
        if (file_error || merges_bytes > 64U * MiB) {
            return make_error(ErrorCode::resource_exhausted, "Tokenizer merges file is missing or too large", merges_path.string());
        }
        std::ifstream merges(merges_path, std::ios::binary);
        if (!merges) return make_error(ErrorCode::io_error, "Unable to open tokenizer merges", merges_path.string());
        std::string line;
        while (std::getline(merges, line)) {
            auto added = add_merge(line);
            if (!added) return added.error();
        }
    }

    util::JsonLimits config_limits;
    config_limits.max_input_bytes = 1U * MiB;
    config_limits.max_nodes = 4096;
    auto config_json = util::load_json_file(config_path, config_limits);
    if (!config_json) return config_json.error();
    if (pending_added_tokens.empty()) {
        const auto* added_decoder = config_json.value().find("added_tokens_decoder");
        if (added_decoder == nullptr) {
            return make_error(ErrorCode::invalid_format, "Tokenizer has no added-token definitions", config_path.string());
        }
        auto added = added_decoder->as_object();
        if (!added || added.value()->size() > 4096U) {
            return make_error(ErrorCode::invalid_format, "Tokenizer added-token map is invalid", config_path.string());
        }
        pending_added_tokens.reserve(added.value()->size());
        for (const auto& [id_text, definition] : *added.value()) {
            std::uint32_t id = 0;
            const auto converted = std::from_chars(id_text.data(), id_text.data() + id_text.size(), id);
            const auto* content_value = definition.find("content");
            const auto* special_value = definition.find("special");
            auto content = content_value == nullptr
                               ? Result<std::string_view>(make_error(ErrorCode::invalid_format, "missing content"))
                               : content_value->as_string();
            auto special = special_value == nullptr
                               ? Result<bool>(make_error(ErrorCode::invalid_format, "missing special"))
                               : special_value->as_bool();
            if (converted.ec != std::errc{} || converted.ptr != id_text.data() + id_text.size() || !content || !special) {
                return make_error(ErrorCode::invalid_format, "Tokenizer added-token entry is invalid", id_text);
            }
            pending_added_tokens.push_back(PendingAddedToken{id, std::string(content.value()), special.value()});
        }
        std::sort(pending_added_tokens.begin(), pending_added_tokens.end(),
                  [](const auto& left, const auto& right) { return left.id < right.id; });
    }
    for (const auto& definition : pending_added_tokens) {
        const std::uint32_t id = definition.id;
        const std::string_view content = definition.content;
        if (id > result.id_to_token_.size()) {
            return make_error(ErrorCode::invalid_format, "Tokenizer added-token ids are not contiguous", std::to_string(id));
        }
        if (id == result.id_to_token_.size()) {
            if (result.token_to_id_.contains(std::string(content))) {
                return make_error(ErrorCode::integrity_error, "Tokenizer appended token duplicates vocabulary", std::to_string(id));
            }
            result.id_to_token_.emplace_back(content);
            result.added_ids_.push_back(false);
            result.special_ids_.push_back(false);
            result.token_to_id_.emplace(result.id_to_token_.back(), id);
        } else if (result.id_to_token_[id] != content) {
            return make_error(ErrorCode::integrity_error, "Tokenizer added token does not match vocabulary", std::to_string(id));
        }
        result.added_tokens_.push_back(SpecialToken{std::string(content), id});
        result.added_ids_[id] = true;
        if (definition.special) result.special_ids_[id] = true;
    }
    std::sort(result.added_tokens_.begin(), result.added_tokens_.end(), [](const auto& left, const auto& right) {
        return left.content.size() > right.content.size();
    });
    result.added_token_trie_.emplace_back();
    for (const auto& token : result.added_tokens_) {
        std::size_t node = 0U;
        for (const char raw_byte : token.content) {
            const auto byte = static_cast<std::uint8_t>(static_cast<unsigned char>(raw_byte));
            auto& edges = result.added_token_trie_[node].edges;
            auto edge = std::lower_bound(edges.begin(), edges.end(), byte,
                                         [](const auto& candidate, std::uint8_t value) {
                                             return candidate.first < value;
                                         });
            if (edge == edges.end() || edge->first != byte) {
                const std::size_t child = result.added_token_trie_.size();
                edges.insert(edge, {byte, child});
                node = child;
                result.added_token_trie_.emplace_back();
            } else {
                node = edge->second;
            }
        }
        if (result.added_token_trie_[node].token_id) {
            return make_error(ErrorCode::integrity_error,
                              "Tokenizer added-token content is duplicated", token.content);
        }
        result.added_token_trie_[node].token_id = token.id;
    }

    const auto assign_token_id = [&](std::string_view field, std::uint32_t* output) -> Result<void> {
        const auto* value = config_json.value().find(field);
        if (value == nullptr) return make_error(ErrorCode::invalid_format, "Tokenizer special-token field is missing", std::string(field));
        auto content = tokenizer_config_token_content(*value, field);
        if (!content) return make_error(ErrorCode::invalid_format, "Tokenizer special-token field is not a string", std::string(field));
        const auto iterator = result.token_to_id_.find(std::string(content.value()));
        if (iterator == result.token_to_id_.end()) {
            return make_error(ErrorCode::integrity_error, "Tokenizer special token is absent from the vocabulary", std::string(field));
        }
        *output = iterator->second;
        return {};
    };
    auto bos = assign_token_id("bos_token", &result.bos_token_id_);
    auto eos = assign_token_id("eos_token", &result.eos_token_id_);
    auto pad = assign_token_id("pad_token", &result.pad_token_id_);
    if (!bos) return bos.error();
    if (!eos) return eos.error();
    if (!pad) return pad.error();

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
        std::optional<std::uint32_t> matched_id;
        std::size_t matched_bytes = 0U;
        std::size_t node = 0U;
        for (std::size_t cursor = position; cursor < text.size();) {
            const auto byte = static_cast<std::uint8_t>(static_cast<unsigned char>(text[cursor]));
            const auto& edges = added_token_trie_[node].edges;
            const auto edge = std::lower_bound(edges.begin(), edges.end(), byte,
                                               [](const auto& candidate, std::uint8_t value) {
                                                   return candidate.first < value;
                                               });
            if (edge == edges.end() || edge->first != byte) break;
            node = edge->second;
            ++cursor;
            if (added_token_trie_[node].token_id) {
                matched_id = added_token_trie_[node].token_id;
                matched_bytes = cursor - position;
            }
        }
        if (!matched_id) {
            ++position;
            continue;
        }
        if (position > ordinary_start) {
            auto encoded = encode_ordinary(text.substr(ordinary_start, position - ordinary_start));
            if (!encoded) return encoded.error();
            output.insert(output.end(), encoded.value().begin(), encoded.value().end());
        }
        output.push_back(*matched_id);
        position += matched_bytes;
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
    if (pretokenizer_mode_ == PretokenizerMode::deepseek_v4) {
        return deepseek_pretokenize(decoded.value());
    }
    std::vector<std::vector<Codepoint>> digit_pieces;
    std::vector<Codepoint> current;
    for (const auto& codepoint : decoded.value()) {
        if (split_digits_individually_ && is_ascii_digit(codepoint)) {
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
        if (added_ids_[id]) {
            if (!special_ids_[id] || !skip_special_tokens) output += id_to_token_[id];
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

bool Gpt2Tokenizer::is_special_token(std::uint32_t id) const noexcept {
    return id < special_ids_.size() && special_ids_[id];
}

Result<std::string> Gpt2Tokenizer::token_bytes(std::uint32_t id) const {
    return decode({id}, false);
}

Result<void> Gpt2Tokenizer::validate_chat_content(std::string_view content) const {
    for (const auto& token : added_tokens_) {
        if (token.id < special_ids_.size() && special_ids_[token.id] &&
            content.find(token.content) != std::string_view::npos) {
            // Deliberately does not echo the content or the token itself: the
            // refusal must not become a probe for reserved markers.
            return make_error(ErrorCode::invalid_argument,
                              "chat message content contains a reserved control token");
        }
    }
    return {};
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
        auto safe = validate_chat_content(message.content);
        if (!safe) {
            return make_error(safe.error().code, safe.error().message, message.role);
        }
        append_message(message.role, message.content);
    }
    if (add_generation_prompt) output += "<|im_start|>assistant\n";
    if (output.size() > kMaximumInputBytes) return make_error(ErrorCode::invalid_argument, "Formatted chat exceeds 16 MiB");
    return output;
}

Result<std::string> Gpt2Tokenizer::apply_olmoe_chat_template(const std::vector<ChatMessage>& messages,
                                                             bool add_generation_prompt) const {
    if (messages.empty() || messages.size() > kMaximumMessages || bos_token_id_ >= id_to_token_.size() ||
        eos_token_id_ >= id_to_token_.size()) {
        return make_error(ErrorCode::invalid_argument, "OLMoE chat requires between 1 and 1024 messages");
    }
    std::string output = id_to_token_[bos_token_id_];
    for (const auto& message : messages) {
        if ((message.role != "system" && message.role != "user" && message.role != "assistant") ||
            message.content.size() > kMaximumInputBytes) {
            return make_error(ErrorCode::invalid_argument, "Invalid OLMoE chat role or oversized message", message.role);
        }
        auto safe = validate_chat_content(message.content);
        if (!safe) {
            return make_error(safe.error().code, safe.error().message, message.role);
        }
        output += "<|";
        output += message.role;
        output += "|>\n";
        output += message.content;
        if (message.role == "assistant") output += id_to_token_[eos_token_id_];
        output.push_back('\n');
    }
    if (add_generation_prompt) output += "<|assistant|>\n";
    if (output.size() > kMaximumInputBytes) return make_error(ErrorCode::invalid_argument, "Formatted OLMoE chat exceeds 16 MiB");
    return output;
}

Result<std::string> Gpt2Tokenizer::apply_deepseek_v4_chat_template(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt) const {
    if (pretokenizer_mode_ != PretokenizerMode::deepseek_v4 || messages.empty() ||
        messages.size() > kMaximumMessages || bos_token_id_ >= id_to_token_.size() ||
        eos_token_id_ >= id_to_token_.size()) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek V4 chat requires its tokenizer and between 1 and 1024 messages");
    }
    const auto user = token_to_id_.find("<｜User｜>");
    const auto assistant = token_to_id_.find("<｜Assistant｜>");
    const auto thinking_end = token_to_id_.find("</think>");
    if (user == token_to_id_.end() || assistant == token_to_id_.end() ||
        thinking_end == token_to_id_.end()) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek V4 tokenizer is missing chat-template tokens");
    }

    std::string output = id_to_token_[bos_token_id_];
    bool saw_system = false;
    bool expect_user = true;
    for (std::size_t index = 0U; index < messages.size(); ++index) {
        const auto& message = messages[index];
        if (message.content.size() > kMaximumInputBytes) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek V4 chat message exceeds 16 MiB", message.role);
        }
        auto safe = validate_chat_content(message.content);
        if (!safe) {
            return make_error(safe.error().code, safe.error().message, message.role);
        }
        if (message.role == "system") {
            if (index != 0U || saw_system) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek V4 system message must be first and unique");
            }
            saw_system = true;
            output += message.content;
            continue;
        }
        if (message.role == "user") {
            if (!expect_user) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek V4 chat roles must alternate user and assistant");
            }
            output += id_to_token_[user->second];
            output += message.content;
            output += id_to_token_[assistant->second];
            output += id_to_token_[thinking_end->second];
            expect_user = false;
            continue;
        }
        if (message.role == "assistant") {
            if (expect_user) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek V4 assistant message must follow a user message");
            }
            output += message.content;
            output += id_to_token_[eos_token_id_];
            expect_user = true;
            continue;
        }
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek V4 basic chat template supports system/user/assistant roles",
                          message.role);
    }
    if (add_generation_prompt) {
        if (expect_user) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek V4 generation prompt requires a final user message");
        }
    } else if (!expect_user) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek V4 completed template requires a final assistant message");
    }
    if (output.size() > kMaximumInputBytes) {
        return make_error(ErrorCode::invalid_argument,
                          "Formatted DeepSeek V4 chat exceeds 16 MiB");
    }
    return output;
}

}  // namespace vastloom::inference

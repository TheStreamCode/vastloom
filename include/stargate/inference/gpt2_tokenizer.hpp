#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "stargate/core/status.hpp"

namespace stargate::inference {

struct ChatMessage {
    std::string role;
    std::string content;
};

class Gpt2Tokenizer {
public:
    [[nodiscard]] static Result<Gpt2Tokenizer> load(const std::filesystem::path& model_directory);

    [[nodiscard]] Result<std::vector<std::uint32_t>> encode(std::string_view text) const;
    [[nodiscard]] Result<std::string> decode(const std::vector<std::uint32_t>& token_ids,
                                             bool skip_special_tokens = true) const;
    [[nodiscard]] Result<std::string> apply_smol_chat_template(const std::vector<ChatMessage>& messages,
                                                               bool add_generation_prompt = true) const;

    [[nodiscard]] std::size_t vocab_size() const noexcept { return id_to_token_.size(); }
    [[nodiscard]] std::uint32_t bos_token_id() const noexcept { return bos_token_id_; }
    [[nodiscard]] std::uint32_t eos_token_id() const noexcept { return eos_token_id_; }
    [[nodiscard]] std::uint32_t pad_token_id() const noexcept { return pad_token_id_; }

private:
    struct SpecialToken {
        std::string content;
        std::uint32_t id{0};
    };

    [[nodiscard]] Result<std::vector<std::uint32_t>> encode_ordinary(std::string_view text) const;
    [[nodiscard]] Result<std::vector<std::string>> pretokenize(std::string_view text) const;
    [[nodiscard]] Result<std::vector<std::uint32_t>> bpe(std::string_view piece) const;

    std::unordered_map<std::string, std::uint32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::size_t> merge_ranks_;
    std::array<std::string, 256> byte_encoder_{};
    std::unordered_map<std::string, std::uint8_t> byte_decoder_;
    std::vector<SpecialToken> special_tokens_;
    std::vector<bool> special_ids_;
    std::uint32_t bos_token_id_{1};
    std::uint32_t eos_token_id_{2};
    std::uint32_t pad_token_id_{2};
};

}  // namespace stargate::inference

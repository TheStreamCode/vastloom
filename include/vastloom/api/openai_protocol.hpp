#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::api {

// One turn of a chat request. Roles are kept as written rather than mapped to an enum: the chat
// template is the checkpoint's, and inventing a role the template does not know would silently
// change the prompt.
struct ChatMessage {
    std::string role;
    std::string content;
};

// What a coding agent actually sends. Everything optional has a bounded default rather than an
// unbounded one, because a request arriving over a socket decides how much work this process does.
struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    std::size_t max_tokens{128U};
    bool stream{false};
    // Absent means the checkpoint's own greedy path. A caller asking for temperature above zero is
    // refused rather than quietly served greedy output, because a sampler this runtime does not
    // implement must not be reported as honoured.
    std::optional<double> temperature;
    std::vector<std::string> stop;
};

struct ChatCompletionLimits {
    ByteCount max_body_bytes{1U * MiB};
    std::size_t max_messages{256U};
    std::size_t max_tokens_ceiling{4'096U};
    std::size_t max_stop_sequences{8U};
};

// Parses an OpenAI chat request, refusing anything outside its bounds rather than clamping it. A
// clamped request is one the caller believes it made and did not.
[[nodiscard]] Result<ChatCompletionRequest> parse_chat_completion_request(
    std::string_view body,
    const ChatCompletionLimits& limits = {});

struct ChatCompletionResponse {
    std::string id;
    std::string model;
    std::string content;
    // OpenAI's vocabulary: "stop" when the model ended by itself, "length" when the budget ran out.
    std::string finish_reason{"stop"};
    std::uint64_t created{0U};
    std::size_t prompt_tokens{0U};
    std::size_t completion_tokens{0U};
};

[[nodiscard]] Result<std::string> serialize_chat_completion_response(
    const ChatCompletionResponse& response);

// One `data:` frame of a streaming response. The first carries the role, each next one a delta, and
// the last an empty delta with the finish reason; the caller closes with `data: [DONE]`.
[[nodiscard]] Result<std::string> serialize_chat_completion_chunk(
    const ChatCompletionResponse& response,
    std::string_view delta,
    bool first,
    bool last);

// The `/v1/models` listing, which is how a client discovers what this server is serving.
[[nodiscard]] Result<std::string> serialize_model_list(std::string_view model_id,
                                                       std::uint64_t created);

// An OpenAI-shaped error body. Clients branch on `type`, so a runtime refusal has to arrive as one
// of these rather than as a bare HTTP status with no body.
[[nodiscard]] std::string serialize_error(std::string_view message,
                                          std::string_view type,
                                          int http_status);

// Maps this project's error codes onto HTTP statuses. The Governor refusing an allocation is a 503
// rather than a 500: the request was well formed and the host was not able, which is a distinction
// a client can act on by retrying later.
[[nodiscard]] int http_status_for(ErrorCode code) noexcept;

}  // namespace vastloom::api

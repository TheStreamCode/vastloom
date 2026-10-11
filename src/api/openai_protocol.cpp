#include "vastloom/api/openai_protocol.hpp"

#include <cmath>
#include <new>
#include <sstream>

#include "vastloom/util/json.hpp"
#include "vastloom/util/string.hpp"

namespace vastloom::api {
namespace {

Result<std::string> required_string(const util::JsonValue::Object& object,
                                    std::string_view key,
                                    std::size_t max_bytes) {
    const auto found = object.find(key);
    if (found == object.end()) {
        return make_error(ErrorCode::invalid_argument, "Chat request is missing a required field",
                          std::string(key));
    }
    auto text = found->second.as_string();
    if (!text) return text.error();
    if (text.value().size() > max_bytes) {
        return make_error(ErrorCode::invalid_argument, "Chat request field exceeds its limit",
                          std::string(key));
    }
    try {
        return std::string(text.value());
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate a chat request field");
    }
}

}  // namespace

StopSequenceFilter::StopSequenceFilter(const std::vector<std::string>& sequences) {
    sequences_.reserve(sequences.size());
    for (const auto& sequence : sequences) {
        if (!sequence.empty()) sequences_.push_back(sequence);
    }
}

std::string_view StopSequenceFilter::feed(std::string_view piece) {
    if (stopped_ || sequences_.empty()) return {};
    pending_.append(piece);

    std::size_t earliest = std::string::npos;
    for (const auto& sequence : sequences_) {
        const auto at = pending_.find(sequence);
        if (at != std::string::npos && (earliest == std::string::npos || at < earliest)) {
            earliest = at;
        }
    }
    if (earliest != std::string::npos) {
        stopped_ = true;
        emitted_.assign(pending_, 0U, earliest);
        pending_.clear();
        return emitted_;
    }

    // Hold back the longest suffix that is still a prefix of some stop sequence: it may complete
    // on the next piece. Everything before it can never be cut and is safe to emit.
    std::size_t holdback = 0U;
    for (const auto& sequence : sequences_) {
        const std::size_t longest = std::min(sequence.size() - 1U, pending_.size());
        for (std::size_t k = longest; k > holdback; --k) {
            if (pending_.compare(pending_.size() - k, k, sequence, 0U, k) == 0) {
                holdback = k;
                break;
            }
        }
    }
    emitted_.assign(pending_, 0U, pending_.size() - holdback);
    pending_.erase(0U, pending_.size() - holdback);
    return emitted_;
}

std::string_view StopSequenceFilter::flush() {
    if (stopped_ || pending_.empty()) return {};
    emitted_ = std::move(pending_);
    pending_.clear();
    return emitted_;
}

std::string StopSequenceFilter::apply(std::string_view text) {
    std::size_t earliest = std::string_view::npos;
    for (const auto& sequence : sequences_) {
        const auto at = text.find(sequence);
        if (at != std::string_view::npos && (earliest == std::string_view::npos || at < earliest)) {
            earliest = at;
        }
    }
    if (earliest == std::string_view::npos) {
        return std::string(text);
    }
    stopped_ = true;
    return std::string(text.substr(0U, earliest));
}

Result<ChatCompletionRequest> parse_chat_completion_request(std::string_view body,
                                                            const ChatCompletionLimits& limits) {
    if (body.size() > limits.max_body_bytes) {
        return make_error(ErrorCode::invalid_argument, "Chat request body exceeds its limit",
                          std::to_string(body.size()));
    }
    util::JsonLimits json_limits;
    json_limits.max_input_bytes = limits.max_body_bytes;
    auto parsed = util::parse_json(body, json_limits);
    if (!parsed) return parsed.error();
    auto root = parsed.value().as_object();
    if (!root) return root.error();
    const auto& object = *root.value();

    try {
        ChatCompletionRequest request;
        auto model = required_string(object, "model", 512U);
        if (!model) return model.error();
        request.model = model.take_value();

        const auto messages = object.find("messages");
        if (messages == object.end()) {
            return make_error(ErrorCode::invalid_argument,
                              "Chat request is missing a required field", "messages");
        }
        auto array = messages->second.as_array();
        if (!array) return array.error();
        if (array.value()->empty()) {
            return make_error(ErrorCode::invalid_argument, "Chat request carries no messages");
        }
        if (array.value()->size() > limits.max_messages) {
            return make_error(ErrorCode::invalid_argument, "Chat request exceeds its message limit",
                              std::to_string(array.value()->size()));
        }
        for (const auto& entry : *array.value()) {
            auto message_object = entry.as_object();
            if (!message_object) return message_object.error();
            auto role = required_string(*message_object.value(), "role", 64U);
            if (!role) return role.error();
            auto content = required_string(*message_object.value(), "content",
                                           static_cast<std::size_t>(limits.max_body_bytes));
            if (!content) return content.error();
            request.messages.push_back(ChatMessage{role.take_value(), content.take_value()});
        }

        const auto read_token_budget = [&](std::string_view key) -> Result<bool> {
            const auto found = object.find(key);
            if (found == object.end() || found->second.is_null()) return false;
            auto value = found->second.as_u64();
            if (!value) return value.error();
            if (value.value() == 0U || value.value() > limits.max_tokens_ceiling) {
                return make_error(ErrorCode::invalid_argument,
                                  "Chat request token budget is outside its limit",
                                  std::to_string(value.value()));
            }
            request.max_tokens = static_cast<std::size_t>(value.value());
            return true;
        };
        // OpenAI renamed this field; accept both and prefer the newer one when both appear.
        auto budget = read_token_budget("max_completion_tokens");
        if (!budget) return budget.error();
        if (!budget.value()) {
            budget = read_token_budget("max_tokens");
            if (!budget) return budget.error();
        }

        if (const auto stream = object.find("stream");
            stream != object.end() && !stream->second.is_null()) {
            auto value = stream->second.as_bool();
            if (!value) return value.error();
            request.stream = value.value();
        }

        if (const auto temperature = object.find("temperature");
            temperature != object.end() && !temperature->second.is_null()) {
            auto value = temperature->second.as_double();
            if (!value) return value.error();
            // A finite double above FLT_MAX would silently become +inf once cast to the
            // sampler's float, so the bound is refused here where the client can see it.
            if (!std::isfinite(value.value()) || value.value() < 0.0 || value.value() > 10.0) {
                return make_error(ErrorCode::invalid_argument,
                                  "Chat request temperature must be between 0 and 10",
                                  std::to_string(value.value()));
            }
            request.temperature = value.value();
        }

        if (const auto stop = object.find("stop");
            stop != object.end() && !stop->second.is_null()) {
            if (stop->second.is_string()) {
                auto value = stop->second.as_string();
                if (!value) return value.error();
                request.stop.emplace_back(value.value());
            } else {
                auto sequences = stop->second.as_array();
                if (!sequences) return sequences.error();
                if (sequences.value()->size() > limits.max_stop_sequences) {
                    return make_error(ErrorCode::invalid_argument,
                                      "Chat request exceeds its stop-sequence limit",
                                      std::to_string(sequences.value()->size()));
                }
                for (const auto& entry : *sequences.value()) {
                    auto value = entry.as_string();
                    if (!value) return value.error();
                    request.stop.emplace_back(value.value());
                }
            }
        }
        return request;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate the parsed chat request");
    }
}

Result<std::string> serialize_chat_completion_response(const ChatCompletionResponse& response) {
    try {
        std::ostringstream out;
        out << "{\"id\":\"" << util::json_escape(response.id)
            << "\",\"object\":\"chat.completion\",\"created\":" << response.created
            << ",\"model\":\"" << util::json_escape(response.model)
            << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
            << util::json_escape(response.content) << "\"},\"finish_reason\":\""
            << util::json_escape(response.finish_reason)
            << "\"}],\"usage\":{\"prompt_tokens\":" << response.prompt_tokens
            << ",\"completion_tokens\":" << response.completion_tokens
            << ",\"total_tokens\":" << response.prompt_tokens + response.completion_tokens
            << "}}";
        return out.str();
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate a chat response");
    }
}

Result<std::string> serialize_chat_completion_chunk(const ChatCompletionResponse& response,
                                                    std::string_view delta,
                                                    bool first,
                                                    bool last) {
    try {
        std::ostringstream out;
        out << "{\"id\":\"" << util::json_escape(response.id)
            << "\",\"object\":\"chat.completion.chunk\",\"created\":" << response.created
            << ",\"model\":\"" << util::json_escape(response.model)
            << "\",\"choices\":[{\"index\":0,\"delta\":{";
        if (first) out << "\"role\":\"assistant\"";
        if (!delta.empty()) {
            if (first) out << ',';
            out << "\"content\":\"" << util::json_escape(delta) << '"';
        }
        out << "},\"finish_reason\":";
        if (last) {
            out << '"' << util::json_escape(response.finish_reason) << '"';
        } else {
            out << "null";
        }
        out << "}]}";
        return out.str();
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate a chat response chunk");
    }
}

Result<std::string> serialize_model_list(std::string_view model_id, std::uint64_t created) {
    try {
        std::ostringstream out;
        out << "{\"object\":\"list\",\"data\":[{\"id\":\"" << util::json_escape(model_id)
            << "\",\"object\":\"model\",\"created\":" << created << ",\"owned_by\":\"vastloom\"}]}";
        return out.str();
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate the model listing");
    }
}

std::string serialize_error(std::string_view message, std::string_view type, int http_status) {
    std::ostringstream out;
    out << "{\"error\":{\"message\":\"" << util::json_escape(message) << "\",\"type\":\""
        << util::json_escape(type) << "\",\"code\":" << http_status << "}}";
    return out.str();
}

int http_status_for(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::invalid_argument:
        case ErrorCode::invalid_format:
            return 400;
        case ErrorCode::permission_denied:
        case ErrorCode::unsafe_path:
            return 403;
        case ErrorCode::not_found:
            return 404;
        case ErrorCode::unsupported:
            return 501;
        // The request was well formed and the host was not able. A client can act on that by
        // retrying later, which it cannot do with a 500.
        case ErrorCode::resource_exhausted:
            return 503;
        case ErrorCode::io_error:
        case ErrorCode::integrity_error:
        case ErrorCode::internal_error:
            break;
    }
    return 500;
}

}  // namespace vastloom::api

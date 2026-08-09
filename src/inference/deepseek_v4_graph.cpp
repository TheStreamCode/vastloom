#include "vastloom/inference/deepseek_v4_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <string>
#include <utility>

namespace vastloom::inference {
namespace {

constexpr std::size_t kMaximumGraphElements = 64U * 1024U * 1024U;
constexpr std::size_t kMaximumReferenceTokens = 4096U;
constexpr std::size_t kMaximumReferenceLayers = 16U;

Result<std::size_t> bounded_product(std::initializer_list<std::size_t> factors,
                                    std::string context) {
    std::size_t result = 1U;
    for (const auto factor : factors) {
        if (factor != 0U && result > std::numeric_limits<std::size_t>::max() / factor) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek reference graph shape overflows", std::move(context));
        }
        result *= factor;
    }
    if (result > kMaximumGraphElements) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek reference graph exceeds its element limit",
                          std::move(context));
    }
    return result;
}

Result<void> finite_values(std::span<const float> values, std::string_view context) {
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek graph tensor contains a non-finite value",
                              std::string(context) + "[" + std::to_string(index) + "]");
        }
    }
    return {};
}

Result<std::span<const float>> float_tensor(const DeepseekV4ReferenceGraph& graph,
                                            std::string_view name,
                                            std::initializer_list<std::size_t> shape) {
    auto expected = bounded_product(shape, std::string(name));
    if (!expected) return expected.error();
    const auto found = graph.float_tensors.find(name);
    if (found == graph.float_tensors.end()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek graph is missing a float tensor", std::string(name));
    }
    if (found->second.size() != expected.value()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek graph tensor shape does not match", std::string(name));
    }
    auto finite = finite_values(found->second, name);
    if (!finite) return finite.error();
    return std::span<const float>(found->second);
}

Result<std::span<const std::uint32_t>> integer_tensor(
    const DeepseekV4ReferenceGraph& graph,
    std::string_view name,
    std::initializer_list<std::size_t> shape) {
    auto expected = bounded_product(shape, std::string(name));
    if (!expected) return expected.error();
    const auto found = graph.integer_tensors.find(name);
    if (found == graph.integer_tensors.end()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek graph is missing an integer tensor", std::string(name));
    }
    if (found->second.size() != expected.value()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek graph integer tensor shape does not match",
                          std::string(name));
    }
    return std::span<const std::uint32_t>(found->second);
}

Result<std::vector<float>> linear(std::span<const float> input,
                                  std::span<const float> weight,
                                  std::size_t rows,
                                  std::size_t input_width,
                                  std::size_t output_width,
                                  std::string_view context) {
    auto input_count = bounded_product({rows, input_width}, std::string(context) + " input");
    if (!input_count) return input_count.error();
    auto weight_count = bounded_product({output_width, input_width},
                                        std::string(context) + " weight");
    if (!weight_count) return weight_count.error();
    auto output_count = bounded_product({rows, output_width}, std::string(context) + " output");
    if (!output_count) return output_count.error();
    if (input.size() != input_count.value() || weight.size() != weight_count.value()) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek graph linear tensor shape does not match",
                          std::string(context));
    }
    std::vector<float> output(output_count.value());
    for (std::size_t row = 0U; row < rows; ++row) {
        for (std::size_t destination = 0U; destination < output_width; ++destination) {
            double sum = 0.0;
            for (std::size_t column = 0U; column < input_width; ++column) {
                sum += static_cast<double>(input[row * input_width + column]) *
                       static_cast<double>(weight[destination * input_width + column]);
            }
            output[row * output_width + destination] = static_cast<float>(sum);
        }
    }
    auto finite = finite_values(output, context);
    if (!finite) return finite.error();
    return output;
}

double rope_frequency(std::size_t pair,
                      std::size_t rope_width,
                      double theta,
                      std::size_t original_context,
                      double factor,
                      double beta_fast,
                      double beta_slow) {
    double frequency = std::pow(theta, -2.0 * static_cast<double>(pair) /
                                           static_cast<double>(rope_width));
    if (original_context == 0U) return frequency;
    const auto correction = [&](double rotations) {
        return static_cast<double>(rope_width) *
               std::log(static_cast<double>(original_context) /
                        (rotations * 2.0 * std::numbers::pi)) /
               (2.0 * std::log(theta));
    };
    const double low = std::max(0.0, std::floor(correction(beta_fast)));
    const double high = std::min(static_cast<double>(rope_width - 1U),
                                 std::ceil(correction(beta_slow)));
    double denominator = high - low;
    if (denominator == 0.0) denominator = 0.001;
    const double ramp = std::clamp((static_cast<double>(pair) - low) / denominator,
                                   0.0, 1.0);
    const double smooth = 1.0 - ramp;
    frequency = frequency / factor * (1.0 - smooth) + frequency * smooth;
    return frequency;
}

Result<void> apply_rope_tail(std::vector<float>& values,
                             std::size_t rows,
                             std::size_t heads,
                             std::size_t head_width,
                             std::size_t rope_width,
                             float theta,
                             std::size_t original_context,
                             float factor,
                             float beta_fast,
                             float beta_slow,
                             std::size_t position_stride,
                             bool inverse = false) {
    if (rope_width == 0U || rope_width > head_width || rope_width % 2U != 0U ||
        !std::isfinite(theta) || theta <= 0.0F || !std::isfinite(factor) || factor <= 0.0F ||
        !std::isfinite(beta_fast) || beta_fast <= 0.0F ||
        !std::isfinite(beta_slow) || beta_slow <= 0.0F) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek graph RoPE configuration is invalid");
    }
    auto expected = bounded_product({rows, heads, head_width}, "DeepSeek graph RoPE values");
    if (!expected) return expected.error();
    if (values.size() != expected.value()) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek graph RoPE value shape does not match");
    }
    const std::size_t rope_offset = head_width - rope_width;
    for (std::size_t row = 0U; row < rows; ++row) {
        const double position = static_cast<double>(row * position_stride);
        for (std::size_t head = 0U; head < heads; ++head) {
            const std::size_t offset = (row * heads + head) * head_width + rope_offset;
            for (std::size_t pair = 0U; pair < rope_width / 2U; ++pair) {
                double angle = position * rope_frequency(
                    pair, rope_width, static_cast<double>(theta), original_context,
                    static_cast<double>(factor), static_cast<double>(beta_fast),
                    static_cast<double>(beta_slow));
                if (inverse) angle = -angle;
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                const double real = static_cast<double>(values[offset + pair * 2U]);
                const double imaginary = static_cast<double>(values[offset + pair * 2U + 1U]);
                values[offset + pair * 2U] =
                    static_cast<float>(real * cosine - imaginary * sine);
                values[offset + pair * 2U + 1U] =
                    static_cast<float>(imaginary * cosine + real * sine);
            }
        }
    }
    return {};
}

Result<void> normalize_attention_queries(std::vector<float>& query,
                                         std::size_t tokens,
                                         std::size_t heads,
                                         std::size_t width,
                                         float epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek query normalization epsilon is invalid");
    }
    for (std::size_t token = 0U; token < tokens; ++token) {
        for (std::size_t head = 0U; head < heads; ++head) {
            const std::size_t offset = (token * heads + head) * width;
            double squares = 0.0;
            for (std::size_t column = 0U; column < width; ++column) {
                const double value = static_cast<double>(query[offset + column]);
                squares += value * value;
            }
            const double inverse = 1.0 / std::sqrt(
                squares / static_cast<double>(width) + static_cast<double>(epsilon));
            for (std::size_t column = 0U; column < width; ++column) {
                query[offset + column] = static_cast<float>(
                    static_cast<double>(query[offset + column]) * inverse);
            }
        }
    }
    return {};
}

struct CompressionResult {
    std::vector<float> values;
    std::size_t count{0};
};

Result<CompressionResult> compress_prefill(const DeepseekV4ReferenceGraph& graph,
                                           std::string_view prefix,
                                           std::span<const float> input,
                                           std::size_t token_count,
                                           std::size_t ratio,
                                           std::size_t output_width,
                                           std::size_t rope_width) {
    const auto& config = graph.config;
    if (ratio == 0U || output_width == 0U || rope_width > output_width) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek compressor configuration is invalid");
    }
    const bool overlap = ratio == 4U;
    const std::size_t coefficient = overlap ? 2U : 1U;
    const std::size_t projected_width = coefficient * output_width;
    const std::string base(prefix);
    auto wkv = float_tensor(graph, base + ".wkv.weight", {projected_width, config.width});
    if (!wkv) return wkv.error();
    auto wgate = float_tensor(graph, base + ".wgate.weight", {projected_width, config.width});
    if (!wgate) return wgate.error();
    auto ape = float_tensor(graph, base + ".ape", {ratio, projected_width});
    if (!ape) return ape.error();
    auto norm = float_tensor(graph, base + ".norm.weight", {output_width});
    if (!norm) return norm.error();
    auto projected = linear(input, wkv.value(), token_count, config.width,
                            projected_width, base + ".wkv");
    if (!projected) return projected.error();
    auto scores = linear(input, wgate.value(), token_count, config.width,
                         projected_width, base + ".wgate");
    if (!scores) return scores.error();
    const std::size_t cutoff = token_count - token_count % ratio;
    CompressionResult result;
    result.count = cutoff / ratio;
    auto result_size = bounded_product({result.count, output_width}, base + " output");
    if (!result_size) return result_size.error();
    result.values.resize(result_size.value());
    for (std::size_t window = 0U; window < result.count; ++window) {
        for (std::size_t dimension = 0U; dimension < output_width; ++dimension) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t candidate = 0U; candidate < coefficient * ratio; ++candidate) {
                const bool previous = overlap && candidate < ratio;
                if (previous && window == 0U) continue;
                const std::size_t position = candidate % ratio;
                const std::size_t source_token =
                    (previous ? window - 1U : window) * ratio + position;
                const std::size_t projected_column =
                    (previous ? 0U : overlap ? output_width : 0U) + dimension;
                const double score = static_cast<double>(
                    scores.value()[source_token * projected_width + projected_column]) +
                    static_cast<double>(ape.value()[position * projected_width + projected_column]);
                maximum = std::max(maximum, score);
            }
            double denominator = 0.0;
            double accumulated = 0.0;
            for (std::size_t candidate = 0U; candidate < coefficient * ratio; ++candidate) {
                const bool previous = overlap && candidate < ratio;
                if (previous && window == 0U) continue;
                const std::size_t position = candidate % ratio;
                const std::size_t source_token =
                    (previous ? window - 1U : window) * ratio + position;
                const std::size_t projected_column =
                    (previous ? 0U : overlap ? output_width : 0U) + dimension;
                const double exponential = std::exp(
                    static_cast<double>(scores.value()[source_token * projected_width + projected_column]) +
                    static_cast<double>(ape.value()[position * projected_width + projected_column]) -
                    maximum);
                denominator += exponential;
                accumulated += exponential * static_cast<double>(
                    projected.value()[source_token * projected_width + projected_column]);
            }
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek compressor normalization is invalid", base);
            }
            result.values[window * output_width + dimension] =
                static_cast<float>(accumulated / denominator);
        }
    }
    if (result.count == 0U) return result;
    auto normalized = deepseek_v4_rms_norm(result.values, norm.value(), result.count,
                                            output_width, config.norm_epsilon);
    if (!normalized) return normalized.error();
    result.values = normalized.take_value();
    auto rope = apply_rope_tail(result.values, result.count, 1U, output_width, rope_width,
                                config.compression_rope_theta, config.original_context,
                                config.rope_factor, config.beta_fast, config.beta_slow,
                                ratio);
    if (!rope) return rope.error();
    return result;
}

Result<std::vector<std::int32_t>> window_indices(std::size_t token_count,
                                                 std::size_t window_size) {
    const std::size_t selected = std::min(token_count, window_size);
    auto count = bounded_product({token_count, selected}, "DeepSeek window indices");
    if (!count) return count.error();
    std::vector<std::int32_t> result(count.value(), -1);
    for (std::size_t token = 0U; token < token_count; ++token) {
        const std::size_t base = token >= window_size ? token - window_size + 1U : 0U;
        for (std::size_t slot = 0U; slot < selected; ++slot) {
            const std::size_t candidate = base + slot;
            if (candidate <= token) {
                result[token * selected + slot] = static_cast<std::int32_t>(candidate);
            }
        }
    }
    return result;
}

Result<std::vector<std::int32_t>> indexer_indices(
    const DeepseekV4ReferenceGraph& graph,
    std::string_view prefix,
    std::span<const float> input,
    std::span<const float> query_rank,
    std::size_t token_count,
    std::size_t ratio,
    std::size_t compressed_count) {
    const auto& config = graph.config;
    const std::string base(prefix);
    auto query_weight = float_tensor(graph, base + ".wq_b.weight",
                                     {config.index_head_count * config.index_head_width,
                                      config.query_lora_rank});
    if (!query_weight) return query_weight.error();
    auto projected_query = linear(query_rank, query_weight.value(), token_count,
                                  config.query_lora_rank,
                                  config.index_head_count * config.index_head_width,
                                  base + ".wq_b");
    if (!projected_query) return projected_query.error();
    auto query_rope = apply_rope_tail(projected_query.value(), token_count,
                                      config.index_head_count, config.index_head_width,
                                      config.rope_width, config.compression_rope_theta,
                                      config.original_context, config.rope_factor,
                                      config.beta_fast, config.beta_slow, 1U);
    if (!query_rope) return query_rope.error();
    auto compressed = compress_prefill(graph, base + ".compressor", input, token_count,
                                       ratio, config.index_head_width, config.rope_width);
    if (!compressed) return compressed.error();
    if (compressed.value().count != compressed_count) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek indexer compression count does not match");
    }
    auto projection_weight = float_tensor(graph, base + ".weights_proj.weight",
                                          {config.index_head_count, config.width});
    if (!projection_weight) return projection_weight.error();
    auto head_weights = linear(input, projection_weight.value(), token_count, config.width,
                               config.index_head_count, base + ".weights_proj");
    if (!head_weights) return head_weights.error();
    const double head_scale = 1.0 / std::sqrt(static_cast<double>(config.index_head_width)) /
                              std::sqrt(static_cast<double>(config.index_head_count));
    const std::size_t selected = std::min(config.index_top_k, compressed_count);
    auto output_count = bounded_product({token_count, selected}, base + " indices");
    if (!output_count) return output_count.error();
    std::vector<std::int32_t> result(output_count.value(), -1);
    std::vector<std::pair<double, std::size_t>> ranking(compressed_count);
    for (std::size_t token = 0U; token < token_count; ++token) {
        const std::size_t allowed = std::min((token + 1U) / ratio, compressed_count);
        for (std::size_t compressed_index = 0U; compressed_index < compressed_count;
             ++compressed_index) {
            double score = -std::numeric_limits<double>::infinity();
            if (compressed_index < allowed) {
                score = 0.0;
                for (std::size_t head = 0U; head < config.index_head_count; ++head) {
                    double dot = 0.0;
                    for (std::size_t column = 0U; column < config.index_head_width; ++column) {
                        dot += static_cast<double>(projected_query.value()[
                                   (token * config.index_head_count + head) *
                                       config.index_head_width + column]) *
                               static_cast<double>(compressed.value().values[
                                   compressed_index * config.index_head_width + column]);
                    }
                    score += std::max(dot, 0.0) *
                             static_cast<double>(head_weights.value()[
                                 token * config.index_head_count + head]) *
                             head_scale;
                }
            }
            ranking[compressed_index] = {score, compressed_index};
        }
        std::partial_sort(ranking.begin(), ranking.begin() + static_cast<std::ptrdiff_t>(selected),
                          ranking.end(), [](const auto& left, const auto& right) {
                              return left.first == right.first ? left.second < right.second
                                                               : left.first > right.first;
                          });
        for (std::size_t slot = 0U; slot < selected; ++slot) {
            if (std::isfinite(ranking[slot].first)) {
                result[token * selected + slot] = static_cast<std::int32_t>(
                    token_count + ranking[slot].second);
            }
        }
    }
    return result;
}

struct AttentionResult {
    std::vector<float> output;
    std::vector<std::int32_t> indices;
    std::size_t selected_per_token{0};
};

Result<AttentionResult> attention_prefill(const DeepseekV4ReferenceGraph& graph,
                                          std::size_t layer,
                                          std::span<const float> input,
                                          std::size_t token_count) {
    const auto& config = graph.config;
    const std::string base = "layers." + std::to_string(layer) + ".attn";
    auto wq_a = float_tensor(graph, base + ".wq_a.weight",
                             {config.query_lora_rank, config.width});
    if (!wq_a) return wq_a.error();
    auto query_rank = linear(input, wq_a.value(), token_count, config.width,
                             config.query_lora_rank, base + ".wq_a");
    if (!query_rank) return query_rank.error();
    auto query_norm_weight = float_tensor(graph, base + ".q_norm.weight",
                                          {config.query_lora_rank});
    if (!query_norm_weight) return query_norm_weight.error();
    auto normalized_query_rank = deepseek_v4_rms_norm(
        query_rank.value(), query_norm_weight.value(), token_count,
        config.query_lora_rank, config.norm_epsilon);
    if (!normalized_query_rank) return normalized_query_rank.error();
    auto wq_b = float_tensor(graph, base + ".wq_b.weight",
                             {config.head_count * config.head_width,
                              config.query_lora_rank});
    if (!wq_b) return wq_b.error();
    auto query = linear(normalized_query_rank.value(), wq_b.value(), token_count,
                        config.query_lora_rank, config.head_count * config.head_width,
                        base + ".wq_b");
    if (!query) return query.error();
    auto query_normalized = normalize_attention_queries(query.value(), token_count,
                                                        config.head_count,
                                                        config.head_width,
                                                        config.norm_epsilon);
    if (!query_normalized) return query_normalized.error();
    const std::size_t ratio = config.compression_ratios[layer];
    const std::size_t original_context = ratio == 0U ? 0U : config.original_context;
    const float theta = ratio == 0U ? config.rope_theta : config.compression_rope_theta;
    auto query_rope = apply_rope_tail(query.value(), token_count, config.head_count,
                                      config.head_width, config.rope_width, theta,
                                      original_context, config.rope_factor,
                                      config.beta_fast, config.beta_slow, 1U);
    if (!query_rope) return query_rope.error();

    auto wkv = float_tensor(graph, base + ".wkv.weight",
                            {config.head_width, config.width});
    if (!wkv) return wkv.error();
    auto key_value = linear(input, wkv.value(), token_count, config.width,
                            config.head_width, base + ".wkv");
    if (!key_value) return key_value.error();
    auto kv_norm_weight = float_tensor(graph, base + ".kv_norm.weight",
                                       {config.head_width});
    if (!kv_norm_weight) return kv_norm_weight.error();
    auto normalized_kv = deepseek_v4_rms_norm(key_value.value(), kv_norm_weight.value(),
                                              token_count, config.head_width,
                                              config.norm_epsilon);
    if (!normalized_kv) return normalized_kv.error();
    key_value = normalized_kv.take_value();
    auto kv_rope = apply_rope_tail(key_value.value(), token_count, 1U, config.head_width,
                                   config.rope_width, theta, original_context,
                                   config.rope_factor, config.beta_fast,
                                   config.beta_slow, 1U);
    if (!kv_rope) return kv_rope.error();

    auto window = window_indices(token_count, config.window_size);
    if (!window) return window.error();
    const std::size_t window_selected = std::min(token_count, config.window_size);
    std::vector<std::int32_t> compressed_indices;
    std::size_t compressed_selected = 0U;
    if (ratio != 0U) {
        const std::size_t compressed_count = token_count / ratio;
        if (ratio == 4U) {
            auto learned = indexer_indices(graph, base + ".indexer", input,
                                           normalized_query_rank.value(), token_count,
                                           ratio, compressed_count);
            if (!learned) return learned.error();
            compressed_indices = learned.take_value();
            compressed_selected = std::min(config.index_top_k, compressed_count);
        } else {
            compressed_selected = compressed_count;
            auto compressed_index_count = bounded_product(
                {token_count, compressed_selected}, base + " compressed indices");
            if (!compressed_index_count) return compressed_index_count.error();
            compressed_indices.assign(compressed_index_count.value(), -1);
            for (std::size_t token = 0U; token < token_count; ++token) {
                const std::size_t allowed = (token + 1U) / ratio;
                for (std::size_t slot = 0U; slot < compressed_selected; ++slot) {
                    if (slot < allowed) {
                        compressed_indices[token * compressed_selected + slot] =
                            static_cast<std::int32_t>(token_count + slot);
                    }
                }
            }
        }
        auto compressed = compress_prefill(graph, base + ".compressor", input,
                                           token_count, ratio, config.head_width,
                                           config.rope_width);
        if (!compressed) return compressed.error();
        key_value.value().insert(key_value.value().end(), compressed.value().values.begin(),
                                 compressed.value().values.end());
    }
    AttentionResult result;
    result.selected_per_token = window_selected + compressed_selected;
    auto index_count = bounded_product({token_count, result.selected_per_token},
                                       base + " attention indices");
    if (!index_count) return index_count.error();
    result.indices.resize(index_count.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        std::copy_n(window.value().begin() + static_cast<std::ptrdiff_t>(token * window_selected),
                    window_selected,
                    result.indices.begin() +
                        static_cast<std::ptrdiff_t>(token * result.selected_per_token));
        if (compressed_selected != 0U) {
            std::copy_n(compressed_indices.begin() +
                            static_cast<std::ptrdiff_t>(token * compressed_selected),
                        compressed_selected,
                        result.indices.begin() + static_cast<std::ptrdiff_t>(
                            token * result.selected_per_token + window_selected));
        }
    }
    auto sink = float_tensor(graph, base + ".attn_sink", {config.head_count});
    if (!sink) return sink.error();
    auto attended = deepseek_v4_sparse_attention(
        query.value(), key_value.value(), sink.value(), result.indices, token_count,
        key_value.value().size() / config.head_width, config.head_count,
        config.head_width, result.selected_per_token,
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(config.head_width))));
    if (!attended) return attended.error();
    auto inverse_rope = apply_rope_tail(attended.value(), token_count, config.head_count,
                                        config.head_width, config.rope_width, theta,
                                        original_context, config.rope_factor,
                                        config.beta_fast, config.beta_slow, 1U, true);
    if (!inverse_rope) return inverse_rope.error();
    const std::size_t group_input =
        config.head_count * config.head_width / config.output_groups;
    auto wo_a = float_tensor(graph, base + ".wo_a.weight",
                             {config.output_groups * config.output_lora_rank, group_input});
    if (!wo_a) return wo_a.error();
    auto group_count = bounded_product(
        {token_count, config.output_groups, config.output_lora_rank}, base + ".wo_a output");
    if (!group_count) return group_count.error();
    std::vector<float> grouped(group_count.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        for (std::size_t group = 0U; group < config.output_groups; ++group) {
            for (std::size_t rank = 0U; rank < config.output_lora_rank; ++rank) {
                double sum = 0.0;
                for (std::size_t column = 0U; column < group_input; ++column) {
                    sum += static_cast<double>(attended.value()[
                               token * config.head_count * config.head_width +
                               group * group_input + column]) *
                           static_cast<double>(wo_a.value()[
                               (group * config.output_lora_rank + rank) * group_input + column]);
                }
                grouped[(token * config.output_groups + group) *
                            config.output_lora_rank + rank] = static_cast<float>(sum);
            }
        }
    }
    auto wo_b = float_tensor(graph, base + ".wo_b.weight",
                             {config.width,
                              config.output_groups * config.output_lora_rank});
    if (!wo_b) return wo_b.error();
    auto projected = linear(grouped, wo_b.value(), token_count,
                            config.output_groups * config.output_lora_rank,
                            config.width, base + ".wo_b");
    if (!projected) return projected.error();
    result.output = projected.take_value();
    return result;
}

Result<std::vector<float>> execute_expert(const DeepseekV4ReferenceGraph& graph,
                                          std::string_view prefix,
                                          std::span<const float> input,
                                          std::span<const float> routing,
                                          std::size_t tokens) {
    const auto& config = graph.config;
    const std::string base(prefix);
    auto gate = float_tensor(graph, base + ".w1.weight",
                             {config.intermediate_width, config.width});
    if (!gate) return gate.error();
    auto down = float_tensor(graph, base + ".w2.weight",
                             {config.width, config.intermediate_width});
    if (!down) return down.error();
    auto up = float_tensor(graph, base + ".w3.weight",
                           {config.intermediate_width, config.width});
    if (!up) return up.error();
    return deepseek_v4_expert_swiglu(input, gate.value(), down.value(), up.value(),
                                     routing, tokens, config.width,
                                     config.intermediate_width, config.swiglu_limit);
}

struct MoeResult {
    std::vector<float> output;
    DeepseekV4RouteResult route;
};

Result<MoeResult> moe_forward(const DeepseekV4ReferenceGraph& graph,
                              std::size_t layer,
                              std::span<const float> input,
                              std::span<const std::uint32_t> token_ids,
                              std::size_t token_count) {
    const auto& config = graph.config;
    const std::string base = "layers." + std::to_string(layer) + ".ffn";
    auto gate_weight = float_tensor(graph, base + ".gate.weight",
                                    {config.expert_count, config.width});
    if (!gate_weight) return gate_weight.error();
    Result<DeepseekV4RouteResult> route = make_error(
        ErrorCode::internal_error, "DeepSeek route was not initialized");
    if (layer < config.hash_layer_count) {
        auto table = integer_tensor(graph, base + ".gate.tid2eid",
                                    {config.vocabulary_size, config.experts_per_token});
        if (!table) return table.error();
        route = deepseek_v4_route_hash(input, gate_weight.value(), token_ids, table.value(),
                                       token_count, config.width, config.vocabulary_size,
                                       config.expert_count, config.experts_per_token,
                                       config.route_scale);
    } else {
        auto bias = float_tensor(graph, base + ".gate.bias", {config.expert_count});
        if (!bias) return bias.error();
        route = deepseek_v4_route_learned(input, gate_weight.value(), bias.value(),
                                          token_count, config.width, config.expert_count,
                                          config.experts_per_token, config.route_scale);
    }
    if (!route) return route.error();
    auto output_count = bounded_product({token_count, config.width}, base + " output");
    if (!output_count) return output_count.error();
    std::vector<float> output(output_count.value(), 0.0F);
    for (std::size_t expert = 0U; expert < config.expert_count; ++expert) {
        for (std::size_t token = 0U; token < token_count; ++token) {
            for (std::size_t slot = 0U; slot < config.experts_per_token; ++slot) {
                if (route.value().indices[token * config.experts_per_token + slot] != expert) {
                    continue;
                }
                const std::span<const float> token_input(
                    input.data() + static_cast<std::ptrdiff_t>(token * config.width),
                    config.width);
                const float route_weight =
                    route.value().weights[token * config.experts_per_token + slot];
                auto expert_output = execute_expert(
                    graph, base + ".experts." + std::to_string(expert), token_input,
                    std::span<const float>(&route_weight, 1U), 1U);
                if (!expert_output) return expert_output.error();
                for (std::size_t column = 0U; column < config.width; ++column) {
                    output[token * config.width + column] += expert_output.value()[column];
                }
            }
        }
    }
    auto shared = execute_expert(graph, base + ".shared_experts", input, {}, token_count);
    if (!shared) return shared.error();
    for (std::size_t index = 0U; index < output.size(); ++index) {
        output[index] += shared.value()[index];
    }
    auto finite = finite_values(output, base + " output");
    if (!finite) return finite.error();
    MoeResult result;
    result.output = std::move(output);
    result.route = route.take_value();
    return result;
}

Result<DeepseekV4HyperConnectionPreResult> hc_pre(
    const DeepseekV4ReferenceGraph& graph,
    std::string_view prefix,
    std::span<const float> input,
    std::size_t token_count) {
    const auto& config = graph.config;
    const std::string base(prefix);
    const std::size_t mix_width = (2U + config.hyper_multiplier) * config.hyper_multiplier;
    auto function = float_tensor(graph, base + "_fn",
                                 {mix_width, config.hyper_multiplier * config.width});
    if (!function) return function.error();
    auto scale = float_tensor(graph, base + "_scale", {3U});
    if (!scale) return scale.error();
    auto bias = float_tensor(graph, base + "_base", {mix_width});
    if (!bias) return bias.error();
    return deepseek_v4_hc_pre(input, function.value(), scale.value(), bias.value(),
                              token_count, config.hyper_multiplier, config.width,
                              config.sinkhorn_iterations, config.norm_epsilon,
                              config.hyper_epsilon);
}

Result<std::vector<float>> hc_head(const DeepseekV4ReferenceGraph& graph,
                                   std::span<const float> input,
                                   std::size_t token_count) {
    const auto& config = graph.config;
    const std::size_t flattened_width = config.hyper_multiplier * config.width;
    auto function = float_tensor(graph, "hc_head_fn",
                                 {config.hyper_multiplier, flattened_width});
    if (!function) return function.error();
    auto scale = float_tensor(graph, "hc_head_scale", {1U});
    if (!scale) return scale.error();
    auto base = float_tensor(graph, "hc_head_base", {config.hyper_multiplier});
    if (!base) return base.error();
    auto mixes = linear(input, function.value(), token_count, flattened_width,
                        config.hyper_multiplier, "DeepSeek HC head");
    if (!mixes) return mixes.error();
    auto output_count = bounded_product({token_count, config.width}, "DeepSeek HC head output");
    if (!output_count) return output_count.error();
    std::vector<float> output(output_count.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        double squares = 0.0;
        for (std::size_t column = 0U; column < flattened_width; ++column) {
            const double value = static_cast<double>(input[token * flattened_width + column]);
            squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(
            squares / static_cast<double>(flattened_width) +
            static_cast<double>(config.norm_epsilon));
        for (std::size_t column = 0U; column < config.width; ++column) {
            double value = 0.0;
            for (std::size_t copy = 0U; copy < config.hyper_multiplier; ++copy) {
                const double mixed = static_cast<double>(
                    mixes.value()[token * config.hyper_multiplier + copy]) * inverse;
                const double sigmoid = 1.0 / (1.0 + std::exp(-(
                    mixed * static_cast<double>(scale.value()[0]) +
                    static_cast<double>(base.value()[copy]))));
                value += (sigmoid + static_cast<double>(config.hyper_epsilon)) *
                         static_cast<double>(input[
                             (token * config.hyper_multiplier + copy) * config.width + column]);
            }
            output[token * config.width + column] = static_cast<float>(value);
        }
    }
    return output;
}

Result<void> validate_graph(const DeepseekV4ReferenceGraph& graph,
                            std::span<const std::uint32_t> token_ids) {
    const auto& config = graph.config;
    if (token_ids.empty() || token_ids.size() > kMaximumReferenceTokens ||
        config.vocabulary_size == 0U || config.width == 0U ||
        config.intermediate_width == 0U || config.layer_count == 0U ||
        config.layer_count > kMaximumReferenceLayers ||
        config.hash_layer_count > config.layer_count || config.head_count == 0U ||
        config.expert_count == 0U || config.experts_per_token == 0U ||
        config.experts_per_token > config.expert_count || config.query_lora_rank == 0U ||
        config.head_width == 0U || config.rope_width == 0U ||
        config.rope_width > config.head_width || config.rope_width % 2U != 0U ||
        config.output_groups == 0U || config.output_lora_rank == 0U ||
        config.window_size == 0U || config.compression_ratios.size() != config.layer_count ||
        config.index_head_count == 0U || config.index_head_width < config.rope_width ||
        config.index_top_k == 0U || config.hyper_multiplier == 0U ||
        config.hyper_multiplier > 16U || config.sinkhorn_iterations == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek reference graph configuration is invalid");
    }
    auto attention_width = bounded_product({config.head_count, config.head_width},
                                           "DeepSeek attention width");
    if (!attention_width) return attention_width.error();
    if (attention_width.value() % config.output_groups != 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek attention width is not divisible by output groups");
    }
    auto indexer_width = bounded_product({config.index_head_count, config.index_head_width},
                                         "DeepSeek indexer width");
    if (!indexer_width) return indexer_width.error();
    auto hyper_width = bounded_product({config.hyper_multiplier, config.width},
                                       "DeepSeek Hyper-Connection width");
    if (!hyper_width) return hyper_width.error();
    auto output_lora_width = bounded_product({config.output_groups,
                                              config.output_lora_rank},
                                             "DeepSeek output LoRA width");
    if (!output_lora_width) return output_lora_width.error();
    auto overlap_width = bounded_product({2U, config.head_width},
                                         "DeepSeek overlap compressor width");
    if (!overlap_width) return overlap_width.error();
    auto overlap_indexer_width = bounded_product({2U, config.index_head_width},
                                                 "DeepSeek overlap indexer width");
    if (!overlap_indexer_width) return overlap_indexer_width.error();
    for (const auto ratio : config.compression_ratios) {
        if (ratio != 0U && ratio != 4U && ratio != 128U) {
            return make_error(ErrorCode::unsupported,
                              "DeepSeek reference graph compression ratio is unsupported",
                              std::to_string(ratio));
        }
    }
    for (const auto value : {config.compression_rope_theta, config.rope_theta,
                             config.rope_factor, config.beta_fast, config.beta_slow,
                             config.norm_epsilon, config.hyper_epsilon,
                             config.route_scale}) {
        if (!std::isfinite(value) || value <= 0.0F) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek reference graph scalar is invalid");
        }
    }
    if (!std::isfinite(config.swiglu_limit) || config.swiglu_limit < 0.0F) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek reference graph SwiGLU clamp is invalid");
    }
    for (const auto token : token_ids) {
        if (token >= config.vocabulary_size) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek reference graph token exceeds the vocabulary",
                              std::to_string(token));
        }
    }
    std::size_t total = 0U;
    for (const auto& [name, values] : graph.float_tensors) {
        if (values.size() > kMaximumGraphElements - total) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek reference graph tensor store exceeds its limit", name);
        }
        total += values.size();
        auto finite = finite_values(values, name);
        if (!finite) return finite.error();
    }
    for (const auto& [name, values] : graph.integer_tensors) {
        if (values.size() > kMaximumGraphElements - total) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek reference graph tensor store exceeds its limit", name);
        }
        total += values.size();
    }
    return {};
}

}  // namespace

Result<DeepseekV4ReferenceForwardResult> deepseek_v4_reference_prefill(
    const DeepseekV4ReferenceGraph& graph,
    std::span<const std::uint32_t> token_ids) {
    auto valid = validate_graph(graph, token_ids);
    if (!valid) return valid.error();
    const auto& config = graph.config;
    const std::size_t token_count = token_ids.size();
    auto embedding = float_tensor(graph, "embed.weight",
                                  {config.vocabulary_size, config.width});
    if (!embedding) return embedding.error();
    auto hidden_count = bounded_product(
        {token_count, config.hyper_multiplier, config.width}, "DeepSeek hidden state");
    if (!hidden_count) return hidden_count.error();
    std::vector<float> hidden(hidden_count.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        for (std::size_t copy = 0U; copy < config.hyper_multiplier; ++copy) {
            std::copy_n(embedding.value().begin() +
                            static_cast<std::ptrdiff_t>(token_ids[token] * config.width),
                        config.width,
                        hidden.begin() + static_cast<std::ptrdiff_t>(
                            (token * config.hyper_multiplier + copy) * config.width));
        }
    }
    DeepseekV4ReferenceForwardResult result;
    result.layers.reserve(config.layer_count);
    for (std::size_t layer = 0U; layer < config.layer_count; ++layer) {
        const std::string layer_base = "layers." + std::to_string(layer);
        const auto residual_attention = hidden;
        auto attention_mix = hc_pre(graph, layer_base + ".hc_attn", hidden, token_count);
        if (!attention_mix) return attention_mix.error();
        auto attention_norm_weight = float_tensor(graph, layer_base + ".attn_norm.weight",
                                                  {config.width});
        if (!attention_norm_weight) return attention_norm_weight.error();
        auto normalized_attention = deepseek_v4_rms_norm(
            attention_mix.value().reduced, attention_norm_weight.value(), token_count,
            config.width, config.norm_epsilon);
        if (!normalized_attention) return normalized_attention.error();
        auto attention = attention_prefill(graph, layer, normalized_attention.value(),
                                           token_count);
        if (!attention) return attention.error();
        auto attention_post = deepseek_v4_hc_post(
            attention.value().output, residual_attention, attention_mix.value().split,
            config.width);
        if (!attention_post) return attention_post.error();
        hidden = attention_post.take_value();

        const auto residual_ffn = hidden;
        auto ffn_mix = hc_pre(graph, layer_base + ".hc_ffn", hidden, token_count);
        if (!ffn_mix) return ffn_mix.error();
        auto ffn_norm_weight = float_tensor(graph, layer_base + ".ffn_norm.weight",
                                            {config.width});
        if (!ffn_norm_weight) return ffn_norm_weight.error();
        auto normalized_ffn = deepseek_v4_rms_norm(
            ffn_mix.value().reduced, ffn_norm_weight.value(), token_count,
            config.width, config.norm_epsilon);
        if (!normalized_ffn) return normalized_ffn.error();
        auto moe = moe_forward(graph, layer, normalized_ffn.value(), token_ids, token_count);
        if (!moe) return moe.error();
        auto ffn_post = deepseek_v4_hc_post(moe.value().output, residual_ffn,
                                            ffn_mix.value().split, config.width);
        if (!ffn_post) return ffn_post.error();
        hidden = ffn_post.take_value();

        DeepseekV4LayerTrace trace;
        trace.layer = layer;
        trace.route = std::move(moe.value().route);
        trace.attention_indices = std::move(attention.value().indices);
        trace.attention_selected_per_token = attention.value().selected_per_token;
        result.layers.push_back(std::move(trace));
    }
    auto reduced = hc_head(graph, hidden, token_count);
    if (!reduced) return reduced.error();
    auto norm_weight = float_tensor(graph, "norm.weight", {config.width});
    if (!norm_weight) return norm_weight.error();
    auto normalized = deepseek_v4_rms_norm(reduced.value(), norm_weight.value(), token_count,
                                           config.width, config.norm_epsilon);
    if (!normalized) return normalized.error();
    auto head = float_tensor(graph, "head.weight", {config.vocabulary_size, config.width});
    if (!head) return head.error();
    const std::span<const float> final_hidden(
        normalized.value().data() +
            static_cast<std::ptrdiff_t>((token_count - 1U) * config.width),
        config.width);
    auto logits = linear(final_hidden, head.value(), 1U, config.width,
                         config.vocabulary_size, "DeepSeek LM head");
    if (!logits) return logits.error();
    result.logits = logits.take_value();
    const auto maximum = std::max_element(result.logits.begin(), result.logits.end());
    result.greedy_token_id = static_cast<std::uint32_t>(
        std::distance(result.logits.begin(), maximum));
    return result;
}

}  // namespace vastloom::inference

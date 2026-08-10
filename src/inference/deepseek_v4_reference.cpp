#include "vastloom/inference/deepseek_v4_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace vastloom::inference {
namespace {

constexpr std::size_t kMaximumReferenceElements = 64U * 1024U * 1024U;

Result<std::size_t> checked_product(std::initializer_list<std::size_t> factors,
                                    std::string context) {
    std::size_t result = 1U;
    for (const auto factor : factors) {
        if (factor != 0U && result > std::numeric_limits<std::size_t>::max() / factor) {
            return make_error(ErrorCode::resource_exhausted,
                              "DeepSeek reference shape overflows", std::move(context));
        }
        result *= factor;
    }
    if (result > kMaximumReferenceElements) {
        return make_error(ErrorCode::resource_exhausted,
                          "DeepSeek reference tensor exceeds the bounded element limit",
                          std::move(context));
    }
    return result;
}

Result<void> require_size(std::size_t actual,
                          std::initializer_list<std::size_t> factors,
                          std::string context) {
    auto expected = checked_product(factors, context);
    if (!expected) return expected.error();
    if (actual != expected.value()) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek reference tensor shape does not match",
                          std::move(context));
    }
    return {};
}

Result<void> require_finite(std::span<const float> values, std::string context) {
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek reference tensor contains a non-finite value",
                              context + "[" + std::to_string(index) + "]");
        }
    }
    return {};
}

Result<void> require_positive_finite(float value, std::string context) {
    if (!std::isfinite(value) || value <= 0.0F) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek reference scalar must be finite and positive",
                          std::move(context));
    }
    return {};
}

float sqrt_softplus(float value) noexcept {
    const double promoted = static_cast<double>(value);
    const double softplus = std::max(promoted, 0.0) + std::log1p(std::exp(-std::abs(promoted)));
    return static_cast<float>(std::sqrt(softplus));
}

float sigmoid(float value) noexcept {
    const double promoted = static_cast<double>(value);
    if (promoted >= 0.0) {
        return static_cast<float>(1.0 / (1.0 + std::exp(-promoted)));
    }
    const double exponential = std::exp(promoted);
    return static_cast<float>(exponential / (1.0 + exponential));
}

Result<DeepseekV4RouteResult> route_from_indices(
    std::span<const float> original_scores,
    std::vector<std::uint32_t> indices,
    std::size_t token_count,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale) {
    if (!std::isfinite(route_scale) || route_scale <= 0.0F) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek route scale must be finite and positive");
    }
    DeepseekV4RouteResult result;
    result.token_count = token_count;
    result.top_k = top_k;
    result.indices = std::move(indices);
    result.weights.resize(token_count * top_k);
    for (std::size_t token = 0U; token < token_count; ++token) {
        double sum = 0.0;
        for (std::size_t slot = 0U; slot < top_k; ++slot) {
            const auto expert = result.indices[token * top_k + slot];
            if (expert >= expert_count) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek route selects an out-of-range expert",
                                  std::to_string(expert));
            }
            const float value = original_scores[token * expert_count + expert];
            result.weights[token * top_k + slot] = value;
            sum += static_cast<double>(value);
        }
        if (!std::isfinite(sum) || sum <= 0.0) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek selected route weights cannot be normalized",
                              std::to_string(token));
        }
        for (std::size_t slot = 0U; slot < top_k; ++slot) {
            result.weights[token * top_k + slot] = static_cast<float>(
                static_cast<double>(result.weights[token * top_k + slot]) / sum *
                static_cast<double>(route_scale));
        }
    }
    return result;
}

Result<std::vector<float>> gate_logits(std::span<const float> input,
                                       std::span<const float> gate_weight,
                                       std::size_t token_count,
                                       std::size_t width,
                                       std::size_t expert_count) {
    if (token_count == 0U || width == 0U || expert_count == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "DeepSeek gate projection dimensions are invalid");
    }
    auto input_size = require_size(input.size(), {token_count, width}, "gate input");
    if (!input_size) return input_size.error();
    auto weight_size = require_size(gate_weight.size(), {expert_count, width}, "gate weight");
    if (!weight_size) return weight_size.error();
    auto input_finite = require_finite(input, "gate input");
    if (!input_finite) return input_finite.error();
    auto weight_finite = require_finite(gate_weight, "gate weight");
    if (!weight_finite) return weight_finite.error();
    auto output_size = checked_product({token_count, expert_count}, "gate logits");
    if (!output_size) return output_size.error();
    std::vector<float> scores(output_size.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        for (std::size_t expert = 0U; expert < expert_count; ++expert) {
            double sum = 0.0;
            for (std::size_t column = 0U; column < width; ++column) {
                sum += static_cast<double>(input[token * width + column]) *
                       static_cast<double>(gate_weight[expert * width + column]);
            }
            scores[token * expert_count + expert] = static_cast<float>(sum);
        }
    }
    return scores;
}

Result<std::vector<float>> activate_gate_logits(std::span<const float> logits,
                                                std::size_t token_count,
                                                std::size_t expert_count) {
    auto size = require_size(logits.size(), {token_count, expert_count}, "gate logits");
    if (!size) return size.error();
    auto finite = require_finite(logits, "gate logits");
    if (!finite) return finite.error();
    std::vector<float> scores(logits.size());
    std::transform(logits.begin(), logits.end(), scores.begin(), sqrt_softplus);
    return scores;
}

}  // namespace

Result<std::vector<float>> deepseek_v4_rms_norm(std::span<const float> input,
                                                std::span<const float> weight,
                                                std::size_t row_count,
                                                 std::size_t width,
                                                 float epsilon) {
    if (row_count == 0U || width == 0U) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek RMSNorm dimensions are invalid");
    }
    auto input_size = require_size(input.size(), {row_count, width}, "RMSNorm input");
    if (!input_size) return input_size.error();
    auto weight_size = require_size(weight.size(), {width}, "RMSNorm weight");
    if (!weight_size) return weight_size.error();
    auto input_finite = require_finite(input, "RMSNorm input");
    if (!input_finite) return input_finite.error();
    auto weight_finite = require_finite(weight, "RMSNorm weight");
    if (!weight_finite) return weight_finite.error();
    auto epsilon_valid = require_positive_finite(epsilon, "RMSNorm epsilon");
    if (!epsilon_valid) return epsilon_valid.error();
    std::vector<float> output(input.size());
    for (std::size_t row = 0U; row < row_count; ++row) {
        double squares = 0.0;
        for (std::size_t column = 0U; column < width; ++column) {
            const double value = static_cast<double>(input[row * width + column]);
            squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(
            squares / static_cast<double>(width) + static_cast<double>(epsilon));
        for (std::size_t column = 0U; column < width; ++column) {
            output[row * width + column] = static_cast<float>(
                static_cast<double>(input[row * width + column]) * inverse *
                static_cast<double>(weight[column]));
        }
    }
    auto output_finite = require_finite(output, "RMSNorm output");
    if (!output_finite) return output_finite.error();
    return output;
}

Result<DeepseekV4RouteResult> deepseek_v4_route_learned(
    std::span<const float> input,
    std::span<const float> gate_weight,
    std::span<const float> selection_bias,
    std::size_t token_count,
    std::size_t width,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale) {
    auto logits = gate_logits(input, gate_weight, token_count, width, expert_count);
    if (!logits) return logits.error();
    return deepseek_v4_route_learned_logits(
        logits.value(), selection_bias, token_count, expert_count, top_k, route_scale);
}

Result<DeepseekV4RouteResult> deepseek_v4_route_learned_logits(
    std::span<const float> logits,
    std::span<const float> selection_bias,
    std::size_t token_count,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale) {
    if (top_k == 0U || top_k > expert_count || expert_count == 0U || token_count == 0U) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek learned route dimensions are invalid");
    }
    auto bias_size = require_size(selection_bias.size(), {expert_count}, "gate selection bias");
    if (!bias_size) return bias_size.error();
    auto bias_finite = require_finite(selection_bias, "gate selection bias");
    if (!bias_finite) return bias_finite.error();
    auto scores = activate_gate_logits(logits, token_count, expert_count);
    if (!scores) return scores.error();
    std::vector<std::uint32_t> indices(token_count * top_k);
    std::vector<std::uint32_t> order(expert_count);
    std::iota(order.begin(), order.end(), 0U);
    for (std::size_t token = 0U; token < token_count; ++token) {
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top_k), order.end(),
                          [&](std::uint32_t left, std::uint32_t right) {
                              const float left_score = scores.value()[token * expert_count + left] + selection_bias[left];
                              const float right_score = scores.value()[token * expert_count + right] + selection_bias[right];
                              return left_score == right_score ? left < right : left_score > right_score;
                          });
        for (std::size_t slot = 0U; slot < top_k; ++slot) {
            indices[token * top_k + slot] = order[slot];
        }
    }
    return route_from_indices(scores.value(), std::move(indices), token_count,
                              expert_count, top_k, route_scale);
}

Result<DeepseekV4RouteResult> deepseek_v4_route_hash(
    std::span<const float> input,
    std::span<const float> gate_weight,
    std::span<const std::uint32_t> token_ids,
    std::span<const std::uint32_t> token_to_experts,
    std::size_t token_count,
    std::size_t width,
    std::size_t vocabulary_size,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale) {
    auto logits = gate_logits(input, gate_weight, token_count, width, expert_count);
    if (!logits) return logits.error();
    return deepseek_v4_route_hash_logits(
        logits.value(), token_ids, token_to_experts, token_count, vocabulary_size,
        expert_count, top_k, route_scale);
}

Result<DeepseekV4RouteResult> deepseek_v4_route_hash_logits(
    std::span<const float> logits,
    std::span<const std::uint32_t> token_ids,
    std::span<const std::uint32_t> token_to_experts,
    std::size_t token_count,
    std::size_t vocabulary_size,
    std::size_t expert_count,
    std::size_t top_k,
    float route_scale) {
    if (top_k == 0U || top_k > expert_count || token_count == 0U ||
        vocabulary_size == 0U || expert_count == 0U) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek hash route dimensions are invalid");
    }
    auto token_size = require_size(token_ids.size(), {token_count}, "hash route token ids");
    if (!token_size) return token_size.error();
    auto map_size = require_size(token_to_experts.size(), {vocabulary_size, top_k}, "hash route table");
    if (!map_size) return map_size.error();
    auto scores = activate_gate_logits(logits, token_count, expert_count);
    if (!scores) return scores.error();
    std::vector<std::uint32_t> indices(token_count * top_k);
    for (std::size_t token = 0U; token < token_count; ++token) {
        if (token_ids[token] >= vocabulary_size) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek hash route token id exceeds its table",
                              std::to_string(token_ids[token]));
        }
        for (std::size_t slot = 0U; slot < top_k; ++slot) {
            indices[token * top_k + slot] = token_to_experts[token_ids[token] * top_k + slot];
        }
    }
    return route_from_indices(scores.value(), std::move(indices), token_count,
                              expert_count, top_k, route_scale);
}

Result<std::vector<float>> deepseek_v4_expert_swiglu(
    std::span<const float> input,
    std::span<const float> gate_weight,
    std::span<const float> down_weight,
    std::span<const float> up_weight,
    std::span<const float> routing_weights,
    std::size_t token_count,
    std::size_t width,
    std::size_t intermediate_width,
    float swiglu_limit) {
    if (token_count == 0U || width == 0U || intermediate_width == 0U ||
        !std::isfinite(swiglu_limit) || swiglu_limit < 0.0F) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek expert dimensions or clamp are invalid");
    }
    const auto validate_tensor = [](std::span<const float> tensor,
                                    std::initializer_list<std::size_t> shape,
                                    const char* context) -> Result<void> {
        auto size = require_size(tensor.size(), shape, context);
        if (!size) return size.error();
        return require_finite(tensor, context);
    };
    auto input_valid = validate_tensor(input, {token_count, width}, "expert input");
    if (!input_valid) return input_valid.error();
    auto gate_valid = validate_tensor(gate_weight, {intermediate_width, width},
                                      "expert gate weight");
    if (!gate_valid) return gate_valid.error();
    auto down_valid = validate_tensor(down_weight, {width, intermediate_width},
                                      "expert down weight");
    if (!down_valid) return down_valid.error();
    auto up_valid = validate_tensor(up_weight, {intermediate_width, width},
                                    "expert up weight");
    if (!up_valid) return up_valid.error();
    auto activated_size = checked_product({token_count, intermediate_width},
                                          "expert activated values");
    if (!activated_size) return activated_size.error();
    if (!routing_weights.empty()) {
        auto route_size = require_size(routing_weights.size(), {token_count}, "expert routing weights");
        if (!route_size) return route_size.error();
        auto route_finite = require_finite(routing_weights, "expert routing weights");
        if (!route_finite) return route_finite.error();
    }
    std::vector<float> activated(activated_size.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        for (std::size_t row = 0U; row < intermediate_width; ++row) {
            double gate = 0.0;
            double up = 0.0;
            for (std::size_t column = 0U; column < width; ++column) {
                const double value = static_cast<double>(input[token * width + column]);
                gate += value * static_cast<double>(gate_weight[row * width + column]);
                up += value * static_cast<double>(up_weight[row * width + column]);
            }
            if (swiglu_limit > 0.0F) {
                gate = std::min(gate, static_cast<double>(swiglu_limit));
                up = std::clamp(up, -static_cast<double>(swiglu_limit),
                                static_cast<double>(swiglu_limit));
            }
            const double silu = gate / (1.0 + std::exp(-gate));
            double value = silu * up;
            if (!routing_weights.empty()) value *= static_cast<double>(routing_weights[token]);
            activated[token * intermediate_width + row] = static_cast<float>(value);
        }
    }
    std::vector<float> output(token_count * width);
    for (std::size_t token = 0U; token < token_count; ++token) {
        for (std::size_t row = 0U; row < width; ++row) {
            double sum = 0.0;
            for (std::size_t column = 0U; column < intermediate_width; ++column) {
                sum += static_cast<double>(activated[token * intermediate_width + column]) *
                       static_cast<double>(down_weight[row * intermediate_width + column]);
            }
            output[token * width + row] = static_cast<float>(sum);
        }
    }
    auto finite = require_finite(output, "expert output");
    if (!finite) return finite.error();
    return output;
}

Result<DeepseekV4HyperConnectionSplit> deepseek_v4_hc_split_sinkhorn(
    std::span<const float> mixes,
    std::span<const float> scale,
    std::span<const float> base,
    std::size_t token_count,
    std::size_t multiplier,
    std::size_t sinkhorn_iterations,
    float epsilon) {
    if (token_count == 0U || multiplier == 0U || multiplier > 16U || sinkhorn_iterations == 0U) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek Hyper-Connection dimensions are invalid");
    }
    auto epsilon_valid = require_positive_finite(epsilon, "Hyper-Connection epsilon");
    if (!epsilon_valid) return epsilon_valid.error();
    const std::size_t mix_width = (2U + multiplier) * multiplier;
    auto mixes_size = require_size(mixes.size(), {token_count, mix_width}, "Hyper-Connection mixes");
    if (!mixes_size) return mixes_size.error();
    auto scale_size = require_size(scale.size(), {3U}, "Hyper-Connection scale");
    if (!scale_size) return scale_size.error();
    auto base_size = require_size(base.size(), {mix_width}, "Hyper-Connection base");
    if (!base_size) return base_size.error();
    for (const auto& [values, name] : {std::pair{mixes, "Hyper-Connection mixes"},
                                      std::pair{scale, "Hyper-Connection scale"},
                                      std::pair{base, "Hyper-Connection base"}}) {
        auto finite = require_finite(values, name);
        if (!finite) return finite.error();
    }
    DeepseekV4HyperConnectionSplit result;
    result.token_count = token_count;
    result.multiplier = multiplier;
    result.pre.resize(token_count * multiplier);
    result.post.resize(token_count * multiplier);
    result.combination.resize(token_count * multiplier * multiplier);
    for (std::size_t token = 0U; token < token_count; ++token) {
        const std::size_t mix_offset = token * mix_width;
        for (std::size_t index = 0U; index < multiplier; ++index) {
            result.pre[token * multiplier + index] =
                sigmoid(mixes[mix_offset + index] * scale[0] + base[index]) + epsilon;
            result.post[token * multiplier + index] =
                2.0F * sigmoid(mixes[mix_offset + multiplier + index] * scale[1] +
                               base[multiplier + index]);
        }
        const std::size_t combination_offset = token * multiplier * multiplier;
        for (std::size_t row = 0U; row < multiplier; ++row) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t column = 0U; column < multiplier; ++column) {
                const std::size_t index = 2U * multiplier + row * multiplier + column;
                const float value = mixes[mix_offset + index] * scale[2] + base[index];
                result.combination[combination_offset + row * multiplier + column] = value;
                maximum = std::max(maximum, value);
            }
            double sum = 0.0;
            for (std::size_t column = 0U; column < multiplier; ++column) {
                auto& value = result.combination[combination_offset + row * multiplier + column];
                value = static_cast<float>(std::exp(static_cast<double>(value - maximum)));
                sum += static_cast<double>(value);
            }
            for (std::size_t column = 0U; column < multiplier; ++column) {
                auto& value = result.combination[combination_offset + row * multiplier + column];
                value = static_cast<float>(static_cast<double>(value) / sum) + epsilon;
            }
        }
        const auto normalize_columns = [&] {
            for (std::size_t column = 0U; column < multiplier; ++column) {
                double sum = 0.0;
                for (std::size_t row = 0U; row < multiplier; ++row) {
                    sum += static_cast<double>(
                        result.combination[combination_offset + row * multiplier + column]);
                }
                for (std::size_t row = 0U; row < multiplier; ++row) {
                    auto& value = result.combination[combination_offset + row * multiplier + column];
                    value = static_cast<float>(
                        static_cast<double>(value) /
                        (sum + static_cast<double>(epsilon)));
                }
            }
        };
        normalize_columns();
        for (std::size_t iteration = 1U; iteration < sinkhorn_iterations; ++iteration) {
            for (std::size_t row = 0U; row < multiplier; ++row) {
                double sum = 0.0;
                for (std::size_t column = 0U; column < multiplier; ++column) {
                    sum += static_cast<double>(
                        result.combination[combination_offset + row * multiplier + column]);
                }
                for (std::size_t column = 0U; column < multiplier; ++column) {
                    auto& value = result.combination[combination_offset + row * multiplier + column];
                    value = static_cast<float>(
                        static_cast<double>(value) /
                        (sum + static_cast<double>(epsilon)));
                }
            }
            normalize_columns();
        }
    }
    return result;
}

Result<DeepseekV4HyperConnectionPreResult> deepseek_v4_hc_pre(
    std::span<const float> input,
    std::span<const float> mixing_weight,
    std::span<const float> scale,
    std::span<const float> base,
    std::size_t token_count,
    std::size_t multiplier,
    std::size_t width,
    std::size_t sinkhorn_iterations,
    float norm_epsilon,
    float hc_epsilon) {
    if (token_count == 0U || multiplier == 0U || multiplier > 16U || width == 0U) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek Hyper-Connection pre dimensions are invalid");
    }
    auto norm_valid = require_positive_finite(norm_epsilon, "Hyper-Connection norm epsilon");
    if (!norm_valid) return norm_valid.error();
    auto flattened_width_result = checked_product({multiplier, width},
                                                  "Hyper-Connection flattened width");
    if (!flattened_width_result) return flattened_width_result.error();
    const std::size_t flattened_width = flattened_width_result.value();
    const std::size_t mix_width = (2U + multiplier) * multiplier;
    auto input_size = require_size(input.size(), {token_count, multiplier, width}, "Hyper-Connection input");
    if (!input_size) return input_size.error();
    auto mixing_size = require_size(mixing_weight.size(), {mix_width, flattened_width}, "Hyper-Connection mixing weight");
    if (!mixing_size) return mixing_size.error();
    auto mixes_size = checked_product({token_count, mix_width}, "Hyper-Connection projected mixes");
    if (!mixes_size) return mixes_size.error();
    auto input_finite = require_finite(input, "Hyper-Connection input");
    if (!input_finite) return input_finite.error();
    auto mixing_finite = require_finite(mixing_weight, "Hyper-Connection mixing weight");
    if (!mixing_finite) return mixing_finite.error();
    std::vector<float> mixes(mixes_size.value());
    for (std::size_t token = 0U; token < token_count; ++token) {
        double squares = 0.0;
        for (std::size_t column = 0U; column < flattened_width; ++column) {
            const double value = static_cast<double>(
                input[token * flattened_width + column]);
            squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(
            squares / static_cast<double>(flattened_width) +
            static_cast<double>(norm_epsilon));
        for (std::size_t row = 0U; row < mix_width; ++row) {
            double sum = 0.0;
            for (std::size_t column = 0U; column < flattened_width; ++column) {
                sum += static_cast<double>(input[token * flattened_width + column]) *
                       static_cast<double>(mixing_weight[row * flattened_width + column]);
            }
            mixes[token * mix_width + row] = static_cast<float>(sum * inverse);
        }
    }
    auto split = deepseek_v4_hc_split_sinkhorn(mixes, scale, base, token_count,
                                                multiplier, sinkhorn_iterations, hc_epsilon);
    if (!split) return split.error();
    DeepseekV4HyperConnectionPreResult result;
    result.width = width;
    result.split = split.take_value();
    result.reduced.resize(token_count * width);
    for (std::size_t token = 0U; token < token_count; ++token) {
        for (std::size_t column = 0U; column < width; ++column) {
            double sum = 0.0;
            for (std::size_t copy = 0U; copy < multiplier; ++copy) {
                sum += static_cast<double>(result.split.pre[token * multiplier + copy]) *
                       static_cast<double>(input[(token * multiplier + copy) * width + column]);
            }
            result.reduced[token * width + column] = static_cast<float>(sum);
        }
    }
    auto reduced_finite = require_finite(result.reduced, "Hyper-Connection reduced output");
    if (!reduced_finite) return reduced_finite.error();
    return result;
}

Result<std::vector<float>> deepseek_v4_hc_post(
    std::span<const float> sublayer,
    std::span<const float> residual,
    const DeepseekV4HyperConnectionSplit& split,
    std::size_t width) {
    if (split.token_count == 0U || split.multiplier == 0U || width == 0U) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek Hyper-Connection post dimensions are invalid");
    }
    auto sublayer_size = require_size(sublayer.size(), {split.token_count, width}, "Hyper-Connection sublayer");
    if (!sublayer_size) return sublayer_size.error();
    auto residual_size = require_size(residual.size(), {split.token_count, split.multiplier, width}, "Hyper-Connection residual");
    if (!residual_size) return residual_size.error();
    auto pre_size = require_size(split.pre.size(), {split.token_count, split.multiplier}, "Hyper-Connection pre weights");
    if (!pre_size) return pre_size.error();
    auto post_size = require_size(split.post.size(), {split.token_count, split.multiplier}, "Hyper-Connection post weights");
    if (!post_size) return post_size.error();
    auto combination_size = require_size(split.combination.size(),
                                         {split.token_count, split.multiplier, split.multiplier},
                                         "Hyper-Connection combination");
    if (!combination_size) return combination_size.error();
    for (const auto& [values, name] : {std::pair{sublayer, "Hyper-Connection sublayer"},
                                      std::pair{residual, "Hyper-Connection residual"},
                                      std::pair{std::span<const float>(split.post), "Hyper-Connection post weights"},
                                      std::pair{std::span<const float>(split.combination), "Hyper-Connection combination"}}) {
        auto finite = require_finite(values, name);
        if (!finite) return finite.error();
    }
    std::vector<float> output(residual.size());
    for (std::size_t token = 0U; token < split.token_count; ++token) {
        for (std::size_t destination = 0U; destination < split.multiplier; ++destination) {
            for (std::size_t column = 0U; column < width; ++column) {
                double value = static_cast<double>(split.post[token * split.multiplier + destination]) *
                               static_cast<double>(sublayer[token * width + column]);
                for (std::size_t source = 0U; source < split.multiplier; ++source) {
                    value += static_cast<double>(split.combination[
                                 (token * split.multiplier + destination) * split.multiplier + source]) *
                             static_cast<double>(residual[(token * split.multiplier + source) * width + column]);
                }
                output[(token * split.multiplier + destination) * width + column] = static_cast<float>(value);
            }
        }
    }
    auto output_finite = require_finite(output, "Hyper-Connection output");
    if (!output_finite) return output_finite.error();
    return output;
}

Result<std::vector<float>> deepseek_v4_apply_rope(
    std::span<const float> input,
    std::span<const std::uint32_t> positions,
    std::size_t row_count,
    std::size_t head_count,
    std::size_t head_width,
    float theta,
    bool inverse) {
    if (row_count == 0U || head_count == 0U || head_width == 0U || head_width % 2U != 0U) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek RoPE dimensions are invalid");
    }
    auto theta_valid = require_positive_finite(theta, "RoPE theta");
    if (!theta_valid) return theta_valid.error();
    auto input_size = require_size(input.size(), {row_count, head_count, head_width}, "RoPE input");
    if (!input_size) return input_size.error();
    auto position_size = require_size(positions.size(), {row_count}, "RoPE positions");
    if (!position_size) return position_size.error();
    auto finite = require_finite(input, "RoPE input");
    if (!finite) return finite.error();
    std::vector<float> output(input.begin(), input.end());
    for (std::size_t row = 0U; row < row_count; ++row) {
        for (std::size_t head = 0U; head < head_count; ++head) {
            const std::size_t offset = (row * head_count + head) * head_width;
            for (std::size_t pair = 0U; pair < head_width / 2U; ++pair) {
                const double frequency = std::pow(static_cast<double>(theta),
                                                  -2.0 * static_cast<double>(pair) /
                                                      static_cast<double>(head_width));
                double angle = static_cast<double>(positions[row]) * frequency;
                if (inverse) angle = -angle;
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                const double real = static_cast<double>(input[offset + pair * 2U]);
                const double imaginary = static_cast<double>(
                    input[offset + pair * 2U + 1U]);
                output[offset + pair * 2U] = static_cast<float>(real * cosine - imaginary * sine);
                output[offset + pair * 2U + 1U] = static_cast<float>(imaginary * cosine + real * sine);
            }
        }
    }
    return output;
}

Result<std::vector<float>> deepseek_v4_sparse_attention(
    std::span<const float> query,
    std::span<const float> key_value,
    std::span<const float> attention_sink,
    std::span<const std::int32_t> selected_indices,
    std::size_t query_count,
    std::size_t key_value_count,
    std::size_t head_count,
    std::size_t head_width,
    std::size_t selected_per_query,
    float softmax_scale) {
    if (query_count == 0U || key_value_count == 0U || head_count == 0U || head_width == 0U ||
        selected_per_query == 0U || !std::isfinite(softmax_scale) || softmax_scale <= 0.0F) {
        return make_error(ErrorCode::invalid_argument, "DeepSeek sparse-attention dimensions are invalid");
    }
    auto query_size = require_size(query.size(), {query_count, head_count, head_width}, "sparse-attention query");
    if (!query_size) return query_size.error();
    auto kv_size = require_size(key_value.size(), {key_value_count, head_width}, "sparse-attention KV");
    if (!kv_size) return kv_size.error();
    auto sink_size = require_size(attention_sink.size(), {head_count}, "attention sink");
    if (!sink_size) return sink_size.error();
    auto indices_size = require_size(selected_indices.size(), {query_count, selected_per_query}, "sparse-attention indices");
    if (!indices_size) return indices_size.error();
    for (const auto& [values, name] : {std::pair{query, "sparse-attention query"},
                                      std::pair{key_value, "sparse-attention KV"},
                                      std::pair{attention_sink, "attention sink"}}) {
        auto finite = require_finite(values, name);
        if (!finite) return finite.error();
    }
    std::vector<float> output(query.size(), 0.0F);
    std::vector<double> scores(selected_per_query);
    for (std::size_t token = 0U; token < query_count; ++token) {
        bool any_valid = false;
        for (std::size_t head = 0U; head < head_count; ++head) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t slot = 0U; slot < selected_per_query; ++slot) {
                const auto selected = selected_indices[token * selected_per_query + slot];
                if (selected == -1) {
                    scores[slot] = -std::numeric_limits<double>::infinity();
                    continue;
                }
                if (selected < 0 || static_cast<std::size_t>(selected) >= key_value_count) {
                    return make_error(ErrorCode::invalid_argument,
                                      "DeepSeek sparse attention selects an invalid KV index",
                                      std::to_string(selected));
                }
                any_valid = true;
                double dot = 0.0;
                for (std::size_t column = 0U; column < head_width; ++column) {
                    dot += static_cast<double>(query[(token * head_count + head) * head_width + column]) *
                           static_cast<double>(key_value[static_cast<std::size_t>(selected) * head_width + column]);
                }
                scores[slot] = dot * static_cast<double>(softmax_scale);
                maximum = std::max(maximum, scores[slot]);
            }
            if (!std::isfinite(maximum)) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek sparse attention query has no selected KV position",
                                  std::to_string(token));
            }
            double denominator = std::exp(static_cast<double>(attention_sink[head]) - maximum);
            for (std::size_t slot = 0U; slot < selected_per_query; ++slot) {
                if (std::isfinite(scores[slot])) denominator += std::exp(scores[slot] - maximum);
            }
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                return make_error(ErrorCode::invalid_argument,
                                  "DeepSeek sparse attention normalization is invalid");
            }
            for (std::size_t column = 0U; column < head_width; ++column) {
                double accumulated = 0.0;
                for (std::size_t slot = 0U; slot < selected_per_query; ++slot) {
                    const auto selected = selected_indices[token * selected_per_query + slot];
                    if (selected < 0) continue;
                    accumulated += std::exp(scores[slot] - maximum) *
                                   static_cast<double>(key_value[static_cast<std::size_t>(selected) * head_width + column]);
                }
                output[(token * head_count + head) * head_width + column] =
                    static_cast<float>(accumulated / denominator);
            }
        }
        if (!any_valid) {
            return make_error(ErrorCode::invalid_argument,
                              "DeepSeek sparse attention query has no valid selected index",
                              std::to_string(token));
        }
    }
    auto finite = require_finite(output, "sparse-attention output");
    if (!finite) return finite.error();
    return output;
}

}  // namespace vastloom::inference

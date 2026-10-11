#include "vastloom/inference/capability_module.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

#include "vastloom/core/bytes.hpp"
#include "vastloom/util/sha256.hpp"

namespace vastloom::inference {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic{'V', 'L', 'C', 'M'};
constexpr std::uint32_t kFormatVersion = 1U;

std::size_t target_payload_bytes(const ModuleTargetSpec& spec, std::uint32_t rank) {
    // A is rank x cols and B is rows x rank, both FP32 little-endian.
    const std::size_t a_elements = static_cast<std::size_t>(rank) * spec.cols;
    const std::size_t b_elements = static_cast<std::size_t>(spec.rows) * rank;
    return (a_elements + b_elements) * sizeof(float);
}

void put_u32(std::vector<std::byte>& out, std::uint32_t value) {
    for (unsigned shift = 0; shift < 4; ++shift) {
        out.push_back(static_cast<std::byte>((value >> (shift * 8U)) & 0xFFU));
    }
}

void put_f32(std::vector<std::byte>& out, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    put_u32(out, bits);
}

bool read_u32(const std::byte* data, std::size_t available, std::size_t& cursor,
              std::uint32_t& value) {
    if (available - cursor < sizeof(std::uint32_t)) return false;
    std::uint32_t bits = 0;
    for (unsigned shift = 0; shift < 4; ++shift) {
        bits |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[cursor + shift]))
                << (shift * 8U);
    }
    cursor += sizeof(std::uint32_t);
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

bool read_f32(const std::byte* data, std::size_t available, std::size_t& cursor, float& value) {
    std::uint32_t bits = 0;
    if (!read_u32(data, available, cursor, bits)) return false;
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

// The loader's failure style: one predicate, one message, no silent clamping anywhere.
Result<void> require(bool condition, std::string message, std::string context = {}) {
    if (!condition) return make_error(ErrorCode::invalid_format, std::move(message), std::move(context));
    return {};
}

[[nodiscard]] bool open_read(const std::filesystem::path& path, FILE** file) noexcept {
#ifdef _WIN32
    return _wfopen_s(file, path.c_str(), L"rb") == 0 && *file != nullptr;
#else
    *file = std::fopen(path.string().c_str(), "rb");
    return *file != nullptr;
#endif
}

}  // namespace

std::string_view to_string(CapabilityTarget target) noexcept {
    switch (target) {
        case CapabilityTarget::q: return "q";
        case CapabilityTarget::k: return "k";
        case CapabilityTarget::v: return "v";
        case CapabilityTarget::o: return "o";
        case CapabilityTarget::gate: return "gate";
        case CapabilityTarget::up: return "up";
        case CapabilityTarget::down: return "down";
        case CapabilityTarget::lm_head: return "lm_head";
        case CapabilityTarget::final_hidden: return "final_hidden";
    }
    return "unknown";
}

std::optional<CapabilityTarget> capability_target_from_u32(std::uint32_t raw) noexcept {
    if (raw <= static_cast<std::uint32_t>(CapabilityTarget::final_hidden)) {
        return static_cast<CapabilityTarget>(raw);
    }
    return std::nullopt;
}

Result<void> write_capability_module(const std::filesystem::path& path,
                                     const CapabilityModule& module) {
    if (const auto checked =
            require(!module.name.empty() && module.name.size() <= 128U,
                    "Capability module name length is outside bounds");
        !checked) {
        return checked.error();
    }
    if (const auto checked = require(module.rank >= 1U && module.rank <= kMaxModuleRank,
                                     "Capability module rank is outside bounds");
        !checked) {
        return checked.error();
    }
    if (const auto checked =
            require(std::isfinite(module.scale) && module.scale >= 0.0F && module.scale <= 1.0e6F,
                    "Capability module scale is not a finite bounded number");
        !checked) {
        return checked.error();
    }
    if (const auto checked =
            require(!module.targets.empty() && module.targets.size() <= kMaxModuleTargets,
                    "Capability module target count is outside bounds");
        !checked) {
        return checked.error();
    }
    const auto finite = [](float value) { return std::isfinite(value); };
    for (const auto& target : module.targets) {
        if (const auto checked =
                require(target.spec.rows >= 1U && target.spec.cols >= 1U,
                        "Capability module target dimensions are empty");
            !checked) {
            return checked.error();
        }
        if (const auto checked =
                require(target.a.size() == static_cast<std::size_t>(module.rank) * target.spec.cols &&
                            target.b.size() ==
                                static_cast<std::size_t>(target.spec.rows) * module.rank,
                        "Capability module adapter storage does not match its dimensions");
            !checked) {
            return checked.error();
        }
        if (const auto checked = require(std::all_of(target.a.begin(), target.a.end(), finite) &&
                                             std::all_of(target.b.begin(), target.b.end(), finite),
                                         "Capability module contains a non-finite weight");
            !checked) {
            return checked.error();
        }
    }
    if (const auto checked =
            require(std::all_of(module.routing_embedding.begin(), module.routing_embedding.end(),
                                finite),
                    "Capability module routing embedding contains a non-finite value");
        !checked) {
        return checked.error();
    }

    std::vector<std::byte> bytes;
    for (const std::uint8_t expected : kMagic) {
        bytes.push_back(static_cast<std::byte>(expected));
    }
    put_u32(bytes, kFormatVersion);
    put_f32(bytes, module.scale);
    put_u32(bytes, module.rank);
    put_u32(bytes, static_cast<std::uint32_t>(module.targets.size()));
    put_u32(bytes, static_cast<std::uint32_t>(kRoutingEmbeddingDim));
    put_u32(bytes, static_cast<std::uint32_t>(module.name.size()));
    for (const char character : module.name) {
        bytes.push_back(static_cast<std::byte>(character));
    }
    put_u32(bytes, 0U);  // reserved

    for (const auto& target : module.targets) {
        put_u32(bytes, target.spec.layer);
        put_u32(bytes, static_cast<std::uint32_t>(target.spec.target));
        put_u32(bytes, target.spec.rows);
        put_u32(bytes, target.spec.cols);
    }
    for (const auto& target : module.targets) {
        for (const float value : target.a) put_f32(bytes, value);
        for (const float value : target.b) put_f32(bytes, value);
    }
    for (const float value : module.routing_embedding) {
        put_f32(bytes, value);
    }

    util::Sha256 hasher;
    hasher.update(std::span<const std::byte>(bytes.data(), bytes.size()));
    for (const std::uint8_t digest_byte : hasher.finalize()) {
        bytes.push_back(static_cast<std::byte>(digest_byte));
    }

    FILE* file = nullptr;
#ifdef _WIN32
    if (_wfopen_s(&file, path.c_str(), L"wb") != 0 || file == nullptr) {
#else
    file = std::fopen(path.string().c_str(), "wb");
    if (file == nullptr) {
#endif
        return make_error(ErrorCode::io_error, "Unable to create capability module", path.string());
    }
    const std::size_t written = std::fwrite(bytes.data(), 1, bytes.size(), file);
    const bool closed = std::fclose(file) == 0;
    if (written != bytes.size() || !closed) {
        return make_error(ErrorCode::io_error, "Short write on capability module", path.string());
    }
    return {};
}

Result<CapabilityModule> load_capability_module(const std::filesystem::path& path) {
    std::error_code size_error;
    const auto raw_size = std::filesystem::file_size(path, size_error);
    if (size_error) {
        return make_error(ErrorCode::not_found, "Unable to stat capability module", path.string());
    }
    if (raw_size > kMaxModuleFileBytes || raw_size < 512U) {
        return make_error(ErrorCode::unsupported, "Capability module size is outside bounds",
                          path.string());
    }

    FILE* file = nullptr;
    if (!open_read(path, &file)) {
        return make_error(ErrorCode::io_error, "Unable to open capability module", path.string());
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(raw_size));
    const std::size_t read = std::fread(bytes.data(), 1, bytes.size(), file);
    std::fclose(file);
    if (read != bytes.size()) {
        return make_error(ErrorCode::io_error, "Short read on capability module", path.string());
    }

    // The trailer is the SHA-256 of everything before it, so any mutation - a flipped weight, a
    // truncated payload, a spliced header - fails here before a single field is trusted.
    constexpr std::size_t kTrailerBytes = 32U;
    const std::size_t body_bytes = bytes.size() - kTrailerBytes;
    util::Sha256 hasher;
    hasher.update(std::span<const std::byte>(bytes.data(), body_bytes));
    const std::array<std::uint8_t, 32> digest = hasher.finalize();
    for (std::size_t index = 0; index < digest.size(); ++index) {
        if (digest[index] != std::to_integer<std::uint8_t>(bytes[body_bytes + index])) {
            return make_error(ErrorCode::integrity_error,
                              "Capability module SHA-256 trailer does not match its contents",
                              path.string());
        }
    }

    const std::byte* data = bytes.data();
    const std::size_t available = body_bytes;
    std::size_t cursor = 0;

    const auto magic_matches = available >= kMagic.size() &&
                               std::equal(kMagic.begin(), kMagic.end(), data,
                                          [](std::uint8_t expected, std::byte actual) {
                                              return expected == std::to_integer<std::uint8_t>(actual);
                                          });
    if (const auto checked = require(magic_matches, "Capability module magic is wrong", path.string());
        !checked) {
        return checked.error();
    }
    cursor += kMagic.size();

    CapabilityModule module;
    std::uint32_t version = 0;
    std::uint32_t rank_raw = 0;
    std::uint32_t target_count = 0;
    std::uint32_t embedding_dim = 0;
    std::uint32_t name_length = 0;
    if (!read_u32(data, available, cursor, version)) {
        return make_error(ErrorCode::invalid_format, "Capability module header is truncated",
                          path.string());
    }
    if (version != kFormatVersion) {
        return make_error(ErrorCode::unsupported, "Capability module version is unsupported",
                          path.string());
    }
    if (!read_f32(data, available, cursor, module.scale) ||
        !read_u32(data, available, cursor, rank_raw) ||
        !read_u32(data, available, cursor, target_count) ||
        !read_u32(data, available, cursor, embedding_dim) ||
        !read_u32(data, available, cursor, name_length)) {
        return make_error(ErrorCode::invalid_format, "Capability module header is truncated",
                          path.string());
    }
    if (const auto checked = require(rank_raw >= 1U && rank_raw <= kMaxModuleRank,
                                     "Capability module rank is outside bounds", path.string());
        !checked) {
        return checked.error();
    }
    module.rank = rank_raw;
    if (const auto checked =
            require(std::isfinite(module.scale) && module.scale >= 0.0F && module.scale <= 1.0e6F,
                    "Capability module scale is not a finite bounded number", path.string());
        !checked) {
        return checked.error();
    }
    if (const auto checked = require(embedding_dim == kRoutingEmbeddingDim,
                                     "Capability module routing embedding has an unexpected width",
                                     path.string());
        !checked) {
        return checked.error();
    }
    if (const auto checked = require(
                name_length >= 1U && name_length <= 128U && available - cursor >= name_length,
                "Capability module name is malformed", path.string());
        !checked) {
        return checked.error();
    }
    module.name.resize(name_length);
    for (std::uint32_t index = 0; index < name_length; ++index) {
        module.name[index] = static_cast<char>(std::to_integer<std::uint8_t>(data[cursor + index]));
    }
    cursor += name_length;
    std::uint32_t reserved = 0;
    if (!read_u32(data, available, cursor, reserved) || reserved != 0U) {
        return make_error(ErrorCode::invalid_format, "Capability module reserved field is not zero",
                          path.string());
    }
    if (const auto checked =
            require(target_count >= 1U && target_count <= kMaxModuleTargets,
                    "Capability module target count is outside bounds", path.string());
        !checked) {
        return checked.error();
    }
    if (const auto checked =
            require(available - cursor >= static_cast<std::size_t>(target_count) * 16U,
                    "Capability module target table is truncated", path.string());
        !checked) {
        return checked.error();
    }

    std::vector<ModuleTargetSpec> specs(target_count);
    std::set<std::pair<std::uint32_t, std::uint32_t>> seen_targets;
    for (std::uint32_t index = 0; index < target_count; ++index) {
        std::uint32_t layer = 0;
        std::uint32_t projection = 0;
        if (!read_u32(data, available, cursor, layer) ||
            !read_u32(data, available, cursor, projection) ||
            !read_u32(data, available, cursor, specs[index].rows) ||
            !read_u32(data, available, cursor, specs[index].cols)) {
            return make_error(ErrorCode::invalid_format, "Capability module target is truncated",
                              path.string());
        }
        const auto parsed = capability_target_from_u32(projection);
        if (!parsed) {
            return make_error(ErrorCode::unsupported, "Capability module names an unknown target",
                              path.string());
        }
        specs[index].layer = layer;
        specs[index].target = parsed.value();
        if (const auto checked = require(specs[index].rows >= 1U && specs[index].cols >= 1U,
                                         "Capability module target dimensions are empty",
                                         path.string());
            !checked) {
            return checked.error();
        }
        if (const auto checked =
                require(seen_targets.emplace(layer, projection).second,
                        "Capability module repeats a layer target", path.string());
            !checked) {
            return checked.error();
        }
    }

    module.targets.reserve(target_count);
    for (std::uint32_t index = 0; index < target_count; ++index) {
        ModuleTarget target;
        target.spec = specs[index];
        target.a.resize(static_cast<std::size_t>(module.rank) * target.spec.cols);
        target.b.resize(static_cast<std::size_t>(target.spec.rows) * module.rank);
        const std::size_t wanted = target_payload_bytes(target.spec, module.rank);
        if (available - cursor < wanted) {
            return make_error(ErrorCode::invalid_format, "Capability module payload is truncated",
                              path.string());
        }
        for (float& value : target.a) {
            if (!read_f32(data, available, cursor, value)) {
                return make_error(ErrorCode::invalid_format,
                                  "Capability module payload ended early", path.string());
            }
        }
        for (float& value : target.b) {
            if (!read_f32(data, available, cursor, value)) {
                return make_error(ErrorCode::invalid_format,
                                  "Capability module payload ended early", path.string());
            }
        }
        const auto all_finite = [](float value) { return std::isfinite(value); };
        if (const auto checked =
                require(std::all_of(target.a.begin(), target.a.end(), all_finite) &&
                            std::all_of(target.b.begin(), target.b.end(), all_finite),
                        "Capability module contains a non-finite weight", path.string());
            !checked) {
            return checked.error();
        }
        module.targets.push_back(std::move(target));
    }
    if (const auto checked =
            require(available - cursor == kRoutingEmbeddingDim * sizeof(float),
                    "Capability module payload length does not match its declaration",
                    path.string());
        !checked) {
        return checked.error();
    }
    for (float& value : module.routing_embedding) {
        if (!read_f32(data, available, cursor, value) || !std::isfinite(value)) {
            return make_error(ErrorCode::invalid_format,
                              "Capability module routing embedding is malformed", path.string());
        }
    }
    if (cursor != available) {
        return make_error(ErrorCode::invalid_format, "Capability module has trailing bytes",
                          path.string());
    }
    module.file_bytes = static_cast<ByteCount>(bytes.size());
    return module;
}

std::array<float, kRoutingEmbeddingDim> embed_request(std::string_view request) {
    // FNV-1a is specified byte-for-byte, unlike std::hash, so the same request embeds the same
    // way on every platform and the router stays reproducible.
    constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
    std::array<float, kRoutingEmbeddingDim> bag{};
    bag.fill(0.0F);

    char previous = 0;
    char before_previous = 0;
    for (const char raw : request) {
        const char character = raw >= 'A' && raw <= 'Z' ? static_cast<char>(raw - 'A' + 'a') : raw;
        if (before_previous != 0 || previous != 0) {
            std::uint64_t hash = kFnvOffset;
            for (const char piece : {before_previous, previous, character}) {
                hash ^= static_cast<std::uint64_t>(static_cast<std::uint8_t>(piece));
                hash *= kFnvPrime;
            }
            const auto bucket = static_cast<std::size_t>((hash >> 8U) % kRoutingEmbeddingDim);
            bag[bucket] += ((hash >> 63U) != 0U) ? 1.0F : -1.0F;
        }
        before_previous = previous;
        previous = character;
    }
    double squares = 0.0;
    for (const float value : bag) {
        squares += static_cast<double>(value) * static_cast<double>(value);
    }
    if (squares <= 0.0) return bag;
    const float norm = static_cast<float>(std::sqrt(squares));
    for (float& value : bag) value /= norm;
    return bag;
}

std::vector<RoutedModule> route_modules(const std::vector<CapabilityModule>& modules,
                                        std::string_view request, std::size_t top_k) {
    const auto query = embed_request(request);
    double query_square = 0.0;
    for (const float value : query) {
        query_square += static_cast<double>(value) * static_cast<double>(value);
    }

    std::vector<RoutedModule> ranked;
    ranked.reserve(modules.size());
    for (std::size_t index = 0; index < modules.size(); ++index) {
        double dot = 0.0;
        double module_square = 0.0;
        for (std::size_t dim = 0; dim < kRoutingEmbeddingDim; ++dim) {
            dot += static_cast<double>(query[dim]) *
                   static_cast<double>(modules[index].routing_embedding[dim]);
            module_square += static_cast<double>(modules[index].routing_embedding[dim]) *
                             static_cast<double>(modules[index].routing_embedding[dim]);
        }
        RoutedModule routed;
        routed.index = index;
        routed.similarity = query_square > 0.0 && module_square > 0.0
                                ? static_cast<float>(dot / std::sqrt(query_square * module_square))
                                : 0.0F;
        ranked.push_back(routed);
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const RoutedModule& left, const RoutedModule& right) {
                         if (left.similarity != right.similarity) {
                             return left.similarity > right.similarity;
                         }
                         return left.index < right.index;
                     });
    if (ranked.size() > top_k) ranked.resize(top_k);
    return ranked;
}

void CapabilityOverlay::add_binding(CapabilityOverlayBinding binding) {
    bindings_.push_back(std::move(binding));
}

const CapabilityOverlayBinding* CapabilityOverlay::find(std::uint32_t layer,
                                                        CapabilityTarget target) const noexcept {
    for (const auto& binding : bindings_) {
        if (binding.layer == layer && binding.target == target) return &binding;
    }
    return nullptr;
}

void CapabilityOverlay::note_macs(std::uint64_t macs) const noexcept { applied_macs_ += macs; }

Result<CapabilityFitResult> fit_lm_head_lora(
    std::size_t vocab_size, std::size_t hidden_size, std::size_t rank, std::size_t steps,
    double learning_rate, double regularization, std::uint64_t seed,
    const std::vector<CapabilityFitPair>& pairs) {
    if (pairs.empty() || pairs.size() > 4'096U) {
        return make_error(ErrorCode::invalid_argument,
                          "LM-head fitting needs between 1 and 4096 supervised positions");
    }
    if (vocab_size < 2U || hidden_size == 0U || rank == 0U || rank > kMaxModuleRank ||
        steps == 0U || steps > 100'000U || !(learning_rate > 0.0) || learning_rate > 10.0 ||
        !std::isfinite(regularization) || regularization < 0.0 || regularization > 1.0e6) {
        return make_error(ErrorCode::invalid_argument,
                          "LM-head fitting bounds are invalid");
    }
    const auto finite = [](double value) { return std::isfinite(value); };
    for (const auto& sample : pairs) {
        if (sample.hidden.size() != hidden_size || sample.base_logits.size() != vocab_size ||
            sample.target_id >= vocab_size) {
            return make_error(ErrorCode::invalid_argument,
                              "A fitting sample does not match the declared dimensions");
        }
        if (!std::all_of(sample.hidden.begin(), sample.hidden.end(), finite) ||
            !std::all_of(sample.base_logits.begin(), sample.base_logits.end(), finite)) {
            return make_error(ErrorCode::invalid_argument,
                              "A fitting sample contains a non-finite value");
        }
    }

    // splitmix64 again: identical initialization on every platform, so a re-run of the same
    // transcript with the same seed reproduces the same module bytes.
    std::uint64_t state = seed;
    const auto next_random = [&state]() {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t mixed = state;
        mixed = (mixed ^ (mixed >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        mixed = (mixed ^ (mixed >> 27U)) * 0x94D049BB133111EBULL;
        return mixed ^ (mixed >> 31U);
    };

    CapabilityFitResult result;
    result.a.resize(rank * hidden_size);
    for (float& value : result.a) {
        value = static_cast<float>(
            static_cast<double>(static_cast<std::int64_t>(next_random() % 2'000'001U) - 1'000'000) / 1.0e10);
    }
    result.b.assign(vocab_size * rank, 0.0F);
    result.positions = pairs.size();

    const double position_scale = learning_rate / static_cast<double>(pairs.size());
    // Decoupled weight decay, AdamW-style: the decay never enters the gradient (an additive
    // L2 term resonated through u and diverged - measured, not guessed). Instead each step
    // ends with a multiplicative contraction clamped at zero, so no hyperparameter pair can
    // overshoot, and an enormous weight simply collapses the adapter back to the frozen base.
    const double decay_factor =
        std::max(0.0, 1.0 - position_scale * regularization);
    std::vector<double> u(rank);
    std::vector<double> grad_b(vocab_size * rank);
    std::vector<double> grad_a(rank * hidden_size);
    std::vector<double> probabilities(vocab_size);

    auto run_step = [&](bool accumulate) {
        double total_loss = 0.0;
        for (const auto& sample : pairs) {
            for (std::size_t r = 0; r < rank; ++r) {
                const float* row = result.a.data() + r * hidden_size;
                double sum = 0.0;
                for (std::size_t h = 0; h < hidden_size; ++h) {
                    sum += static_cast<double>(row[h]) *
                           static_cast<double>(sample.hidden[h]);
                }
                u[r] = sum;
            }
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < vocab_size; ++i) {
                const float* row = result.b.data() + i * rank;
                double delta = 0.0;
                for (std::size_t r = 0; r < rank; ++r) {
                    delta += static_cast<double>(row[r]) * u[r];
                }
                const double z = static_cast<double>(sample.base_logits[i]) + delta;
                probabilities[i] = z;
                if (z > maximum) maximum = z;
            }
            double denominator = 0.0;
            for (std::size_t i = 0; i < vocab_size; ++i) {
                probabilities[i] = std::exp(probabilities[i] - maximum);
                denominator += probabilities[i];
            }
            total_loss += -std::log(probabilities[sample.target_id] / denominator);
            if (!accumulate) continue;
            for (std::size_t i = 0; i < vocab_size; ++i) {
                const double g = probabilities[i] / denominator -
                                 (i == sample.target_id ? 1.0 : 0.0);
                if (g == 0.0) continue;
                double* grad_row = grad_b.data() + i * rank;
                for (std::size_t r = 0; r < rank; ++r) grad_row[r] += g * u[r];
            }
            for (std::size_t r = 0; r < rank; ++r) {
                double du = 0.0;
                const float* b_column = result.b.data() + r;
                for (std::size_t i = 0; i < vocab_size; ++i) {
                    du += static_cast<double>(b_column[i * rank]) *
                          (probabilities[i] / denominator -
                           (i == sample.target_id ? 1.0 : 0.0));
                }
                if (du == 0.0) continue;
                double* grad_row = grad_a.data() + r * hidden_size;
                for (std::size_t h = 0; h < hidden_size; ++h) {
                    grad_row[h] += du * static_cast<double>(sample.hidden[h]);
                }
            }
        }
        return total_loss / static_cast<double>(pairs.size());
    };

    result.initial_loss = run_step(false);
    for (std::size_t step = 0; step < steps; ++step) {
        std::fill(grad_b.begin(), grad_b.end(), 0.0);
        std::fill(grad_a.begin(), grad_a.end(), 0.0);
        static_cast<void>(run_step(true));
        bool diverged = false;
        for (std::size_t index = 0; index < result.a.size() && !diverged; ++index) {
            const float updated = static_cast<float>(
                static_cast<double>(result.a[index]) - position_scale * grad_a[index]);
            if (!std::isfinite(updated)) diverged = true;
            else result.a[index] = updated;
        }
        for (std::size_t index = 0; index < result.b.size() && !diverged; ++index) {
            const float updated = static_cast<float>(
                static_cast<double>(result.b[index]) - position_scale * grad_b[index]);
            if (!std::isfinite(updated)) diverged = true;
            else result.b[index] = updated;
        }
        if (diverged) {
            return make_error(ErrorCode::resource_exhausted,
                              "LM-head fitting diverged to a non-finite weight",
                              "step " + std::to_string(step));
        }
        if (decay_factor != 1.0) {
            for (float& value : result.a) {
                value = static_cast<float>(static_cast<double>(value) * decay_factor);
            }
            for (float& value : result.b) {
                value = static_cast<float>(static_cast<double>(value) * decay_factor);
            }
        }
    }
    result.final_loss = run_step(false);
    return result;
}

Result<CapabilityFitResult> fit_final_hidden_lora(
    std::size_t vocab_size, std::size_t hidden_size, std::size_t rank, std::size_t steps,
    double learning_rate, double regularization, std::uint64_t seed,
    std::span<const float> head_weights,
    const std::vector<HiddenFitSample>& samples) {
    if (samples.empty() || samples.size() > 4'096U) {
        return make_error(ErrorCode::invalid_argument,
                          "Final-hidden fitting needs between 1 and 4096 supervised positions");
    }
    if (vocab_size < 2U || hidden_size == 0U || rank == 0U || rank > kMaxModuleRank ||
        steps == 0U || steps > 100'000U || !(learning_rate > 0.0) || learning_rate > 10.0 ||
        !std::isfinite(regularization) || regularization < 0.0 || regularization > 1.0e6) {
        return make_error(ErrorCode::invalid_argument,
                          "Final-hidden fitting bounds are invalid");
    }
    if (head_weights.size() != vocab_size * hidden_size) {
        return make_error(ErrorCode::invalid_argument,
                          "Head weight matrix does not match the declared dimensions");
    }
    const auto finite = [](double value) { return std::isfinite(value); };
    for (const auto& sample : samples) {
        if (sample.hidden.size() != hidden_size || sample.target_id >= vocab_size ||
            !std::all_of(sample.hidden.begin(), sample.hidden.end(), finite)) {
            return make_error(ErrorCode::invalid_argument,
                              "A fitting sample does not match the declared dimensions");
        }
    }

    // splitmix64 again: identical initialization on every platform.
    std::uint64_t state = seed;
    const auto next_random = [&state]() {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t mixed = state;
        mixed = (mixed ^ (mixed >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        mixed = (mixed ^ (mixed >> 27U)) * 0x94D049BB133111EBULL;
        return mixed ^ (mixed >> 31U);
    };

    CapabilityFitResult result;
    // Unlike the LM-head fitter, here the projection sits inside a double multiplication
    // chain (u = A h feeds both the forward delta and its own gradient), so a tiny init
    // attenuates the learning signal by u^2 and training stalls flat - measured. B still
    // starts at zero, which alone preserves the parity-before-capability start.
    result.a.resize(rank * hidden_size);
    for (float& value : result.a) {
        value = static_cast<float>(
            static_cast<double>(static_cast<std::int64_t>(next_random() % 2'000'001U) - 1'000'000) / 2.0e7);
    }
    // B maps the rank bottleneck back into hidden space: rows = hidden, cols = rank.
    result.b.assign(hidden_size * rank, 0.0F);
    result.positions = samples.size();

    // Base logits are adapter-independent: one head sweep up front removes them from every
    // later inner loop.
    std::vector<std::vector<float>> base_logits(samples.size());
    for (std::size_t s = 0; s < samples.size(); ++s) {
        base_logits[s].resize(vocab_size);
        const float* sample_hidden = samples[s].hidden.data();
        for (std::size_t i = 0; i < vocab_size; ++i) {
            const float* row = head_weights.data() + i * hidden_size;
            double sum = 0.0;
            for (std::size_t h = 0; h < hidden_size; ++h) {
                sum += static_cast<double>(row[h]) *
                       static_cast<double>(sample_hidden[h]);
            }
            if (!std::isfinite(sum)) {
                return make_error(ErrorCode::invalid_argument,
                                  "Head product produced a non-finite logit");
            }
            base_logits[s][i] = static_cast<float>(sum);
        }
    }

    const double position_scale =
        learning_rate / static_cast<double>(samples.size());
    const double decay_factor =
        std::max(0.0, 1.0 - position_scale * regularization);
    // The projection A stays frozen at its deterministic random initialization and only B
    // trains. Measured fact: letting both factors move creates an A<->B feedback loop through
    // ||W*B|| that diverges within tens of steps at any learning rate, clipping included -
    // freezing A removes the resonance term entirely while keeping the gradients closed-form.
    // G = W*B is rebuilt once per step: forward deltas become rank-wide dots against it, and
    // the gradient folds back through the same factor in a second head-sized sweep - two
    // passes per step regardless of how many supervised positions ride on them.
    std::vector<double> g_matrix(vocab_size * rank);
    std::vector<double> grad_g(vocab_size * rank);
    std::vector<double> grad_b(hidden_size * rank);
    std::vector<double> probabilities(vocab_size);
    std::vector<std::vector<double>> fixed_u(samples.size(), std::vector<double>(rank));
    for (std::size_t s = 0; s < samples.size(); ++s) {
        for (std::size_t r = 0; r < rank; ++r) {
            const float* row = result.a.data() + r * hidden_size;
            double sum = 0.0;
            for (std::size_t h = 0; h < hidden_size; ++h) {
                sum += static_cast<double>(row[h]) *
                       static_cast<double>(samples[s].hidden[h]);
            }
            fixed_u[s][r] = sum;
        }
    }

    auto rebuild_g_matrix = [&] {
        for (std::size_t i = 0; i < vocab_size; ++i) {
            const float* w_row = head_weights.data() + i * hidden_size;
            double* g_row = g_matrix.data() + i * rank;
            for (std::size_t r = 0; r < rank; ++r) g_row[r] = 0.0;
            for (std::size_t h = 0; h < hidden_size; ++h) {
                const double weight = static_cast<double>(w_row[h]);
                if (weight == 0.0) continue;
                const float* b_row = result.b.data() + h * rank;
                for (std::size_t r = 0; r < rank; ++r) {
                    g_row[r] += weight * static_cast<double>(b_row[r]);
                }
            }
        }
    };

    auto run_step = [&](bool accumulate) {
        rebuild_g_matrix();
        double total_loss = 0.0;
        if (accumulate) {
            std::fill(grad_g.begin(), grad_g.end(), 0.0);
            std::fill(grad_b.begin(), grad_b.end(), 0.0);
        }
        for (std::size_t s = 0; s < samples.size(); ++s) {
            const auto& sample = samples[s];
            const auto& base = base_logits[s];
            const auto& sample_u = fixed_u[s];
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < vocab_size; ++i) {
                const double* g_row = g_matrix.data() + i * rank;
                double delta = 0.0;
                for (std::size_t r = 0; r < rank; ++r) delta += g_row[r] * sample_u[r];
                const double z = static_cast<double>(base[i]) + delta;
                probabilities[i] = z;
                if (z > maximum) maximum = z;
            }
            double denominator = 0.0;
            for (std::size_t i = 0; i < vocab_size; ++i) {
                probabilities[i] = std::exp(probabilities[i] - maximum);
                denominator += probabilities[i];
            }
            total_loss += -std::log(probabilities[sample.target_id] / denominator);
            if (!accumulate) continue;
            for (std::size_t i = 0; i < vocab_size; ++i) {
                const double g = probabilities[i] / denominator -
                                 (i == sample.target_id ? 1.0 : 0.0);
                if (g == 0.0) continue;
                double* grad_g_row = grad_g.data() + i * rank;
                for (std::size_t r = 0; r < rank; ++r) grad_g_row[r] += g * sample_u[r];
            }
        }
        if (accumulate) {
            // dL/dB[h][r] = sum_i W[i][h] * dL/dG[i][r]: the second head-sized sweep.
            for (std::size_t i = 0; i < vocab_size; ++i) {
                const float* w_row = head_weights.data() + i * hidden_size;
                const double* dg_row = grad_g.data() + i * rank;
                for (std::size_t h = 0; h < hidden_size; ++h) {
                    const double weight = static_cast<double>(w_row[h]);
                    if (weight == 0.0) continue;
                    double* grad_row = grad_b.data() + h * rank;
                    for (std::size_t r = 0; r < rank; ++r) grad_row[r] += weight * dg_row[r];
                }
            }
        }
        return total_loss / static_cast<double>(samples.size());
    };

    result.initial_loss = run_step(false);
    // Two measured stabilizers stack here. Freezing the projection A removes the A<->B
    // resonance that diverged within tens of steps when both factors trained; the global-norm
    // clip additionally bounds every B step while preserving its direction, because the
    // head-transpose path still amplifies gradients through 49k vocabulary rows.
    constexpr double kMaxGradientNorm = 1.0;
    for (std::size_t step = 0; step < steps; ++step) {
        std::fill(grad_b.begin(), grad_b.end(), 0.0);
        static_cast<void>(run_step(true));
        double squared_norm = 0.0;
        for (const double value : grad_b) squared_norm += value * value;
        const double clip = squared_norm > kMaxGradientNorm * kMaxGradientNorm
                                ? kMaxGradientNorm / std::sqrt(squared_norm)
                                : 1.0;
        bool diverged = false;
        for (std::size_t index = 0; index < result.b.size() && !diverged; ++index) {
            const float updated = static_cast<float>(
                static_cast<double>(result.b[index]) - position_scale * clip * grad_b[index]);
            if (!std::isfinite(updated)) diverged = true;
            else result.b[index] = updated;
        }
        if (diverged) {
            return make_error(ErrorCode::resource_exhausted,
                              "Final-hidden fitting diverged to a non-finite weight",
                              "step " + std::to_string(step));
        }
        if (decay_factor != 1.0) {
            for (float& value : result.b) {
                value = static_cast<float>(static_cast<double>(value) * decay_factor);
            }
        }
    }
    result.final_loss = run_step(false);
    return result;
}

Result<void> apply_capability_delta(const CapabilityOverlayBinding& binding,
                                    std::span<const float> input,
                                    std::span<float> hidden_scratch,
                                    std::span<float> output) {
    if (const std::size_t rank = binding.rank;
        input.size() != binding.cols || hidden_scratch.size() < rank || output.size() < binding.rows) {
        return make_error(ErrorCode::invalid_argument,
                          "Capability delta buffers do not match the binding's dimensions",
                          std::string(to_string(binding.target)));
    }
    // Pass one: h = A @ x in double, narrowed once. Pass two: y += scale * (B @ h), also in
    // double per row, matching how the scalar reference matvec accumulates its own rows.
    for (std::size_t r = 0; r < binding.rank; ++r) {
        const float* row = binding.a + r * binding.cols;
        double sum = 0.0;
        for (std::size_t c = 0; c < binding.cols; ++c) {
            sum += static_cast<double>(row[c]) * static_cast<double>(input[c]);
        }
        hidden_scratch[r] = static_cast<float>(sum);
    }
    for (std::size_t i = 0; i < binding.rows; ++i) {
        const float* row = binding.b + i * binding.rank;
        double sum = 0.0;
        for (std::size_t r = 0; r < binding.rank; ++r) {
            sum += static_cast<double>(row[r]) * static_cast<double>(hidden_scratch[r]);
        }
        output[i] += static_cast<float>(static_cast<double>(binding.scale) * sum);
    }
    return {};
}

}  // namespace vastloom::inference

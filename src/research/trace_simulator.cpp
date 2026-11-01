#include "vastloom/research/trace_simulator.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <deque>
#include <fstream>
#include <list>
#include <limits>
#include <map>
#include <numeric>
#include <new>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vastloom/util/string.hpp"

namespace vastloom::research {
namespace {

struct TraceRecord {
    std::uint64_t token{0};
    std::uint32_t layer{0};
    std::uint32_t expert{0};
    ByteCount size{0};
    bool has_readiness_timing{false};
    double submitted_ms{0.0};
    double queue_ms{0.0};
    double ready_ms{0.0};
    double compute_started_ms{0.0};
    double compute_completed_ms{0.0};
    std::uint32_t route_order{0};
    std::uint32_t reduction_order{0};
};

using ExpertKey = std::uint64_t;

ExpertKey make_key(std::uint32_t layer, std::uint32_t expert) noexcept {
    return (static_cast<ExpertKey>(layer) << 32U) | static_cast<ExpertKey>(expert);
}

Result<std::uint64_t> parse_u64(std::string_view value, std::string_view field) {
    const std::string normalized = util::trim(value);
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(normalized.data(), normalized.data() + normalized.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != normalized.data() + normalized.size()) {
        return make_error(ErrorCode::invalid_format, "Invalid trace integer", std::string(field));
    }
    return parsed;
}

Result<double> parse_nonnegative_double(std::string_view value, std::string_view field) {
    const std::string normalized = util::trim(value);
    double parsed = 0.0;
    const auto result = std::from_chars(normalized.data(), normalized.data() + normalized.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != normalized.data() + normalized.size() ||
        !std::isfinite(parsed) || parsed < 0.0) {
        return make_error(ErrorCode::invalid_format, "Invalid nonnegative trace number", std::string(field));
    }
    return parsed;
}

Result<std::vector<TraceRecord>> load_trace(const std::filesystem::path& path, std::size_t max_records) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        return make_error(ErrorCode::not_found, "Unable to stat trace file", path.string());
    }
    if (size > 2ULL * GiB) {
        return make_error(ErrorCode::invalid_format, "Trace file exceeds the 2 GiB safety limit", path.string());
    }
    std::ifstream input(path);
    if (!input) {
        return make_error(ErrorCode::io_error, "Unable to open trace file", path.string());
    }

    std::vector<TraceRecord> records;
    records.reserve(std::min<std::size_t>(max_records, 1'000'000));
    std::string line;
    std::size_t line_number = 0;
    bool header_seen = false;
    std::optional<std::size_t> record_width;
    std::unordered_map<ExpertKey, ByteCount> sizes;
    while (std::getline(input, line)) {
        ++line_number;
        const auto trimmed = util::trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        const auto fields = util::split(trimmed, ',', true);
        if (fields.size() != 4U && fields.size() != 9U && fields.size() != 12U && fields.size() != 19U) {
            return make_error(ErrorCode::invalid_format,
                              "Trace rows require 4 baseline, 9 legacy, 12 prefetch or 19 readiness fields",
                              "line " + std::to_string(line_number));
        }
        if (util::to_lower(util::trim(fields[0])) == "token") {
            if (header_seen || !records.empty() || util::to_lower(util::trim(fields[1])) != "layer" ||
                util::to_lower(util::trim(fields[2])) != "expert" ||
                util::to_lower(util::trim(fields[3])) != "size_bytes") {
                return make_error(ErrorCode::invalid_format, "Invalid or repeated trace header", "line " + std::to_string(line_number));
            }
            if (fields.size() == 19U) {
                constexpr std::array<std::string_view, 19> expected{
                    "token", "layer", "expert", "size_bytes", "routing_weight", "tier", "hit",
                    "bytes_read", "io_ms", "prefetched", "prefetch_wait_ms", "prefetch_overlap_ms",
                    "submitted_ms", "queue_ms", "ready_ms", "compute_started_ms",
                    "compute_completed_ms", "route_order", "reduction_order",
                };
                for (std::size_t index = 0U; index < expected.size(); ++index) {
                    if (util::to_lower(util::trim(fields[index])) != expected[index]) {
                        return make_error(ErrorCode::invalid_format, "Invalid readiness trace header",
                                          "line " + std::to_string(line_number));
                    }
                }
            }
            header_seen = true;
            record_width = fields.size();
            continue;
        }
        if (record_width && *record_width != fields.size()) {
            return make_error(ErrorCode::invalid_format, "Trace rows do not match the declared format",
                              "line " + std::to_string(line_number));
        }
        if (!record_width) record_width = fields.size();
        if (records.size() >= max_records) {
            return make_error(ErrorCode::resource_exhausted, "Trace exceeds configured record limit", path.string());
        }

        auto token = parse_u64(fields[0], "token"); if (!token) return token.error();
        auto layer = parse_u64(fields[1], "layer"); if (!layer || layer.value() > std::numeric_limits<std::uint32_t>::max()) return make_error(ErrorCode::invalid_format, "Layer is outside uint32 range", "line " + std::to_string(line_number));
        auto expert = parse_u64(fields[2], "expert"); if (!expert || expert.value() > std::numeric_limits<std::uint32_t>::max()) return make_error(ErrorCode::invalid_format, "Expert is outside uint32 range", "line " + std::to_string(line_number));
        auto record_size = parse_bytes(fields[3]); if (!record_size || record_size.value() == 0) return make_error(ErrorCode::invalid_format, "Expert size must be greater than zero", "line " + std::to_string(line_number));
        TraceRecord record{token.take_value(), static_cast<std::uint32_t>(layer.take_value()),
                           static_cast<std::uint32_t>(expert.take_value()), record_size.take_value()};
        if (fields.size() >= 9U) {
            auto routing_weight = parse_nonnegative_double(fields[4], "routing_weight");
            const auto tier = util::to_lower(util::trim(fields[5]));
            auto hit = parse_u64(fields[6], "hit");
            auto bytes_read = parse_bytes(fields[7]);
            auto io_ms = parse_nonnegative_double(fields[8], "io_ms");
            if (!routing_weight || routing_weight.value() > 1.0 ||
                (tier != "nvme" && tier != "ram" && tier != "vram") ||
                !hit || hit.value() > 1U || !bytes_read || !io_ms ||
                (hit.value() == 1U && bytes_read.value() != 0U) ||
                (hit.value() == 0U && bytes_read.value() == 0U)) {
                return make_error(ErrorCode::invalid_format, "Invalid extended telemetry fields", "line " + std::to_string(line_number));
            }
            if (fields.size() == 12U) {
                auto prefetched = parse_u64(fields[9], "prefetched");
                auto prefetch_wait_ms = parse_nonnegative_double(fields[10], "prefetch_wait_ms");
                auto prefetch_overlap_ms = parse_nonnegative_double(fields[11], "prefetch_overlap_ms");
                if (!prefetched || prefetched.value() > 1U || !prefetch_wait_ms || !prefetch_overlap_ms ||
                    (hit.value() == 1U && prefetched.value() != 0U)) {
                    return make_error(ErrorCode::invalid_format, "Invalid prefetch telemetry fields",
                                      "line " + std::to_string(line_number));
                }
            }
            if (fields.size() == 19U) {
                auto prefetched = parse_u64(fields[9], "prefetched");
                auto prefetch_wait_ms = parse_nonnegative_double(fields[10], "prefetch_wait_ms");
                auto prefetch_overlap_ms = parse_nonnegative_double(fields[11], "prefetch_overlap_ms");
                auto submitted_ms = parse_nonnegative_double(fields[12], "submitted_ms");
                auto queue_ms = parse_nonnegative_double(fields[13], "queue_ms");
                auto ready_ms = parse_nonnegative_double(fields[14], "ready_ms");
                auto compute_started_ms = parse_nonnegative_double(fields[15], "compute_started_ms");
                auto compute_completed_ms = parse_nonnegative_double(fields[16], "compute_completed_ms");
                auto route_order = parse_u64(fields[17], "route_order");
                auto reduction_order = parse_u64(fields[18], "reduction_order");
                if (!prefetched || prefetched.value() > 1U || !prefetch_wait_ms || !prefetch_overlap_ms ||
                    !submitted_ms || !queue_ms || !ready_ms || !compute_started_ms ||
                    !compute_completed_ms || !route_order || !reduction_order ||
                    route_order.value() > std::numeric_limits<std::uint32_t>::max() ||
                    reduction_order.value() > std::numeric_limits<std::uint32_t>::max() ||
                    ready_ms.value() < submitted_ms.value() ||
                    compute_completed_ms.value() < compute_started_ms.value() ||
                    queue_ms.value() > ready_ms.value() - submitted_ms.value() + 0.01 ||
                    (hit.value() == 1U && prefetched.value() != 0U)) {
                    return make_error(ErrorCode::invalid_format, "Invalid readiness telemetry fields",
                                      "line " + std::to_string(line_number));
                }
                record.has_readiness_timing = true;
                record.submitted_ms = submitted_ms.value();
                record.queue_ms = queue_ms.value();
                record.ready_ms = ready_ms.value();
                record.compute_started_ms = compute_started_ms.value();
                record.compute_completed_ms = compute_completed_ms.value();
                record.route_order = static_cast<std::uint32_t>(route_order.value());
                record.reduction_order = static_cast<std::uint32_t>(reduction_order.value());
            }
        }

        const ExpertKey key = make_key(record.layer, record.expert);
        const auto [iterator, inserted] = sizes.emplace(key, record.size);
        if (!inserted && iterator->second != record.size) {
            return make_error(ErrorCode::integrity_error, "The same expert has inconsistent sizes in the trace", "line " + std::to_string(line_number));
        }
        records.push_back(record);
    }
    if (!input.eof()) {
        return make_error(ErrorCode::io_error, "I/O error while reading trace", path.string());
    }
    if (records.empty()) {
        return make_error(ErrorCode::invalid_format, "Trace contains no records", path.string());
    }
    return records;
}

class LruCache {
public:
    explicit LruCache(ByteCount capacity) : capacity_(capacity) {}

    bool contains(ExpertKey key) const { return entries_.contains(key); }

    ByteCount size_of(ExpertKey key) const {
        const auto iterator = entries_.find(key);
        return iterator == entries_.end() ? 0 : iterator->second.size;
    }

    void touch(ExpertKey key) {
        const auto iterator = entries_.find(key);
        if (iterator == entries_.end()) return;
        order_.splice(order_.begin(), order_, iterator->second.position);
        iterator->second.position = order_.begin();
    }

    bool erase(ExpertKey key) {
        const auto iterator = entries_.find(key);
        if (iterator == entries_.end()) return false;
        used_ -= iterator->second.size;
        order_.erase(iterator->second.position);
        entries_.erase(iterator);
        return true;
    }

    std::vector<std::pair<ExpertKey, ByteCount>> insert(ExpertKey key, ByteCount size) {
        std::vector<std::pair<ExpertKey, ByteCount>> evicted;
        if (capacity_ == 0 || size > capacity_) return evicted;
        if (contains(key)) {
            touch(key);
            return evicted;
        }
        while (!order_.empty() && used_ > capacity_ - size) {
            const ExpertKey victim = order_.back();
            const auto iterator = entries_.find(victim);
            evicted.emplace_back(victim, iterator->second.size);
            used_ -= iterator->second.size;
            order_.pop_back();
            entries_.erase(iterator);
        }
        order_.push_front(key);
        entries_.emplace(key, Entry{size, order_.begin()});
        used_ += size;
        return evicted;
    }

private:
    struct Entry {
        ByteCount size;
        std::list<ExpertKey>::iterator position;
    };

    ByteCount capacity_{0};
    ByteCount used_{0};
    std::list<ExpertKey> order_;
    std::unordered_map<ExpertKey, Entry> entries_;
};

class LfuCache {
public:
    explicit LfuCache(ByteCount capacity) : capacity_(capacity) {}

    bool contains(ExpertKey key) const { return entries_.contains(key); }

    void touch(ExpertKey key) {
        const auto iterator = entries_.find(key);
        if (iterator == entries_.end()) return;
        ranking_.erase(ranking_key(iterator->first, iterator->second));
        ++iterator->second.frequency;
        iterator->second.last_use = ++clock_;
        ranking_.emplace(iterator->second.frequency, iterator->second.last_use, key);
    }

    bool erase(ExpertKey key) {
        const auto iterator = entries_.find(key);
        if (iterator == entries_.end()) return false;
        used_ -= iterator->second.size;
        ranking_.erase(ranking_key(iterator->first, iterator->second));
        entries_.erase(iterator);
        return true;
    }

    std::vector<std::pair<ExpertKey, ByteCount>> insert(ExpertKey key, ByteCount size) {
        std::vector<std::pair<ExpertKey, ByteCount>> evicted;
        if (capacity_ == 0U || size > capacity_) return evicted;
        if (contains(key)) {
            touch(key);
            return evicted;
        }
        // The ordered index keeps the least (frequency, last_use) entry at the front, so an
        // eviction is O(log n) instead of a full scan per victim on million-record traces.
        while (!entries_.empty() && used_ > capacity_ - size) {
            const ExpertKey victim = std::get<2>(*ranking_.begin());
            const auto iterator = entries_.find(victim);
            evicted.emplace_back(victim, iterator->second.size);
            used_ -= iterator->second.size;
            ranking_.erase(ranking_.begin());
            entries_.erase(iterator);
        }
        entries_.emplace(key, Entry{size, 1U, ++clock_});
        ranking_.emplace(1U, clock_, key);
        used_ += size;
        return evicted;
    }

private:
    struct Entry {
        ByteCount size{0U};
        std::uint64_t frequency{0U};
        std::uint64_t last_use{0U};
    };

    static std::tuple<std::uint64_t, std::uint64_t, ExpertKey> ranking_key(
        ExpertKey key, const Entry& entry) noexcept {
        return {entry.frequency, entry.last_use, key};
    }

    ByteCount capacity_{0U};
    ByteCount used_{0U};
    std::uint64_t clock_{0U};
    std::unordered_map<ExpertKey, Entry> entries_;
    std::set<std::tuple<std::uint64_t, std::uint64_t, ExpertKey>> ranking_;
};

// Bounded two-hit admission, mirroring AdaptiveExpertAdmission in
// inference/adaptive_admission.hpp. The simulator keeps its own copy for the same reason
// it keeps its own LRU/LFU policies: it replays traces without runtime cache state.
class GhostHistory {
public:
    explicit GhostHistory(std::size_t limit) : limit_(limit == 0U ? 1U : limit) {}

    [[nodiscard]] bool should_admit(ExpertKey key) {
        if (seen_.contains(key)) return true;
        if (order_.size() == limit_) {
            seen_.erase(order_.front());
            order_.pop_front();
        }
        order_.push_back(key);
        seen_.insert(key);
        return false;
    }

private:
    std::size_t limit_;
    std::deque<ExpertKey> order_;
    std::unordered_set<ExpertKey> seen_;
};

// Entry count the tier capacity buys at the trace's mean expert size, clamped to the
// native 4..1024 slots before the ghost multiplier is applied.
std::size_t admission_ghost_limit(const std::vector<TraceRecord>& records,
                                  const TraceSimulationConfig& config) {
    long double total_bytes = 0.0L;
    for (const auto& record : records) {
        total_bytes += static_cast<long double>(record.size);
    }
    if (records.empty() || total_bytes <= 0.0L) return 16U;
    const long double mean_bytes = total_bytes / static_cast<long double>(records.size());
    const long double capacity = static_cast<long double>(config.vram_cache_bytes) +
                                 static_cast<long double>(config.ram_cache_bytes);
    const long double slots = std::clamp(capacity / mean_bytes, 4.0L, 1024.0L);
    return static_cast<std::size_t>(slots) * 4U;
}

template <typename Value>
Value percentile_sorted(const std::vector<Value>& values, double quantile) {
    if (values.empty()) return 0;
    const auto index = static_cast<std::size_t>(std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1U)];
}

class FenwickTree {
public:
    explicit FenwickTree(std::size_t size) : tree_(size + 1U, 0) {}

    void add(std::size_t position, std::int64_t delta) {
        for (std::size_t index = position; index < tree_.size(); index += index & (~index + 1U)) {
            tree_[index] += delta;
        }
    }

    [[nodiscard]] std::size_t prefix_sum(std::size_t position) const {
        std::int64_t result = 0;
        for (std::size_t index = position; index > 0U; index -= index & (~index + 1U)) {
            result += tree_[index];
        }
        return static_cast<std::size_t>(result);
    }

private:
    std::vector<std::int64_t> tree_;
};

void populate_reuse_metrics(const std::vector<TraceRecord>& records, TraceSimulationReport& report) {
    FenwickTree recent_positions(records.size());
    std::unordered_map<ExpertKey, std::size_t> last_position;
    last_position.reserve(report.unique_experts);
    std::vector<std::size_t> distances;
    distances.reserve(records.size() > report.unique_experts ? records.size() - report.unique_experts : 0U);

    for (std::size_t index = 0U; index < records.size(); ++index) {
        const std::size_t position = index + 1U;
        const ExpertKey key = make_key(records[index].layer, records[index].expert);
        const auto found = last_position.find(key);
        if (found == last_position.end()) {
            ++report.cold_expert_accesses;
        } else {
            const std::size_t active = last_position.size();
            distances.push_back(active - recent_positions.prefix_sum(found->second));
            recent_positions.add(found->second, -1);
            ++report.reused_expert_accesses;
        }
        recent_positions.add(position, 1);
        last_position[key] = position;
    }

    if (report.records != 0U) {
        report.reuse_access_ratio = static_cast<double>(report.reused_expert_accesses) /
                                    static_cast<double>(report.records);
    }
    if (!distances.empty()) {
        const long double total = std::accumulate(distances.begin(), distances.end(), 0.0L);
        report.average_unique_reuse_distance = static_cast<double>(total / static_cast<long double>(distances.size()));
        std::sort(distances.begin(), distances.end());
        report.p50_unique_reuse_distance = percentile_sorted(distances, 0.50);
        report.p95_unique_reuse_distance = percentile_sorted(distances, 0.95);
        report.maximum_unique_reuse_distance = distances.back();
    }
}

Result<void> populate_ready_order_metrics(const std::vector<TraceRecord>& records,
                                          TraceSimulationReport& report) {
    if (records.empty() || !records.front().has_readiness_timing) return {};

    using GroupKey = std::pair<std::uint64_t, std::uint32_t>;
    std::map<GroupKey, std::vector<const TraceRecord*>> groups;
    for (const auto& record : records) {
        if (!record.has_readiness_timing) {
            return make_error(ErrorCode::invalid_format, "Trace mixes readiness and legacy records");
        }
        auto& group = groups[{record.token, record.layer}];
        if (group.size() >= 1'024U) {
            return make_error(ErrorCode::resource_exhausted,
                              "Ready-order group exceeds the 1024-expert safety limit");
        }
        group.push_back(&record);
    }

    constexpr double kTimingEpsilonMs = 1e-6;
    for (auto& [key, group] : groups) {
        (void)key;
        std::sort(group.begin(), group.end(), [](const TraceRecord* left, const TraceRecord* right) {
            return left->route_order < right->route_order;
        });
        std::vector<bool> reduction_seen(group.size(), false);
        for (std::size_t index = 0U; index < group.size(); ++index) {
            if (group[index]->route_order != index || group[index]->reduction_order >= group.size() ||
                reduction_seen[group[index]->reduction_order]) {
                return make_error(ErrorCode::invalid_format,
                                  "Ready-order trace requires unique contiguous route and reduction orders");
            }
            reduction_seen[group[index]->reduction_order] = true;
        }

        double fixed_clock = 0.0;
        for (const TraceRecord* record : group) {
            const double wait = std::max(0.0, record->ready_ms - fixed_clock);
            report.fixed_order_wait_ms += wait;
            fixed_clock += wait + (record->compute_completed_ms - record->compute_started_ms);
        }
        report.fixed_order_makespan_ms += fixed_clock;

        double ready_clock = 0.0;
        std::vector<bool> completed(group.size(), false);
        for (std::size_t step = 0U; step < group.size(); ++step) {
            std::optional<std::size_t> selected;
            for (std::size_t index = 0U; index < group.size(); ++index) {
                if (completed[index] || group[index]->ready_ms > ready_clock + kTimingEpsilonMs) continue;
                if (!selected || group[index]->route_order < group[*selected]->route_order) selected = index;
            }
            if (!selected) {
                for (std::size_t index = 0U; index < group.size(); ++index) {
                    if (completed[index]) continue;
                    if (!selected ||
                        std::pair{group[index]->ready_ms, group[index]->route_order} <
                            std::pair{group[*selected]->ready_ms, group[*selected]->route_order}) {
                        selected = index;
                    }
                }
                const double wait = std::max(0.0, group[*selected]->ready_ms - ready_clock);
                report.ready_order_wait_ms += wait;
                ready_clock += wait;
            }
            const TraceRecord* record = group[*selected];
            if (record->route_order != step) ++report.ready_order_reordered_experts;
            ready_clock += record->compute_completed_ms - record->compute_started_ms;
            completed[*selected] = true;
        }
        report.ready_order_makespan_ms += ready_clock;
    }

    report.ready_order_available = true;
    report.ready_order_groups = groups.size();
    report.ready_order_saved_ms = report.fixed_order_makespan_ms - report.ready_order_makespan_ms;
    if (report.fixed_order_makespan_ms > 0.0) {
        report.ready_order_savings_ratio = report.ready_order_saved_ms / report.fixed_order_makespan_ms;
    }
    return {};
}

void finalize_report(TraceSimulationReport& report, const std::map<std::uint64_t, ByteCount>& per_token) {
    report.tokens = per_token.size();
    const double requests = static_cast<double>(report.records);
    if (requests > 0.0) {
        report.vram_hit_ratio = static_cast<double>(report.vram_hits) / requests;
        report.ram_hit_ratio = static_cast<double>(report.ram_hits) / requests;
        report.storage_miss_ratio = static_cast<double>(report.storage_misses) / requests;
    }
    std::vector<ByteCount> values;
    values.reserve(per_token.size());
    for (const auto& [token, bytes] : per_token) {
        (void)token;
        values.push_back(bytes);
    }
    if (!values.empty()) {
        const long double total = std::accumulate(values.begin(), values.end(), 0.0L);
        report.average_nvme_bytes_per_token = static_cast<double>(total / static_cast<long double>(values.size()));
        std::sort(values.begin(), values.end());
        report.p50_nvme_bytes_per_token = percentile_sorted(values, 0.50);
        report.p95_nvme_bytes_per_token = percentile_sorted(values, 0.95);
        report.maximum_nvme_bytes_per_token = values.back();
    }
}

TraceSimulationReport simulate_lru(const std::vector<TraceRecord>& records, const TraceSimulationConfig& config) {
    TraceSimulationReport report;
    report.policy = CachePolicy::lru;
    report.records = records.size();
    LruCache vram(config.vram_cache_bytes);
    LruCache ram(config.ram_cache_bytes);
    std::unordered_set<ExpertKey> unique;
    std::map<std::uint64_t, ByteCount> per_token;

    for (const auto& record : records) {
        const ExpertKey key = make_key(record.layer, record.expert);
        unique.insert(key);
        per_token.try_emplace(record.token, 0);
        if (vram.contains(key)) {
            ++report.vram_hits;
            vram.touch(key);
            continue;
        }

        if (ram.contains(key)) {
            ++report.ram_hits;
            ram.erase(key);
            if (config.vram_cache_bytes != 0) {
                const auto evicted = vram.insert(key, record.size);
                if (vram.contains(key)) {
                    report.pcie_bytes += record.size;
                    report.evictions += evicted.size();
                    for (const auto& [victim, size] : evicted) {
                        const auto ram_evicted = ram.insert(victim, size);
                        report.evictions += ram_evicted.size();
                    }
                } else {
                    // An expert too large for the VRAM tier stays resident in the RAM tier
                    // instead of silently becoming a permanent storage miss.
                    ram.insert(key, record.size);
                }
            } else {
                ram.insert(key, record.size);
            }
            continue;
        }

        ++report.storage_misses;
        report.nvme_bytes += record.size;
        per_token[record.token] += record.size;
        if (config.vram_cache_bytes != 0) {
            const auto evicted = vram.insert(key, record.size);
            if (vram.contains(key)) {
                report.pcie_bytes += record.size;
                report.evictions += evicted.size();
                for (const auto& [victim, size] : evicted) {
                    const auto ram_evicted = ram.insert(victim, size);
                    report.evictions += ram_evicted.size();
                }
            } else {
                // An expert too large for the VRAM tier belongs in the RAM tier: a
                // declined insert must not turn into a permanent storage miss.
                const auto ram_evicted = ram.insert(key, record.size);
                report.evictions += ram_evicted.size();
            }
        } else {
            const auto ram_evicted = ram.insert(key, record.size);
            report.evictions += ram_evicted.size();
        }
    }
    report.unique_experts = unique.size();
    finalize_report(report, per_token);
    return report;
}

TraceSimulationReport simulate_lfu(const std::vector<TraceRecord>& records, const TraceSimulationConfig& config) {
    TraceSimulationReport report;
    report.policy = CachePolicy::lfu;
    report.records = records.size();
    LfuCache vram(config.vram_cache_bytes);
    LfuCache ram(config.ram_cache_bytes);
    std::unordered_set<ExpertKey> unique;
    std::map<std::uint64_t, ByteCount> per_token;
    for (const auto& record : records) {
        const ExpertKey key = make_key(record.layer, record.expert);
        unique.insert(key);
        per_token.try_emplace(record.token, 0U);
        if (vram.contains(key)) {
            ++report.vram_hits;
            vram.touch(key);
            continue;
        }
        if (ram.contains(key)) {
            ++report.ram_hits;
            if (config.vram_cache_bytes != 0U) {
                ram.erase(key);
                const auto evicted = vram.insert(key, record.size);
                if (vram.contains(key)) {
                    report.pcie_bytes += record.size;
                    report.evictions += evicted.size();
                    for (const auto& [victim, size] : evicted) {
                        const auto ram_evicted = ram.insert(victim, size);
                        report.evictions += ram_evicted.size();
                    }
                } else {
                    // An expert too large for the VRAM tier stays resident in the RAM tier.
                    ram.insert(key, record.size);
                }
            } else {
                ram.touch(key);
            }
            continue;
        }
        ++report.storage_misses;
        report.nvme_bytes += record.size;
        per_token[record.token] += record.size;
        if (config.vram_cache_bytes != 0U) {
            const auto evicted = vram.insert(key, record.size);
            if (vram.contains(key)) {
                report.pcie_bytes += record.size;
                report.evictions += evicted.size();
                for (const auto& [victim, size] : evicted) {
                    const auto ram_evicted = ram.insert(victim, size);
                    report.evictions += ram_evicted.size();
                }
            } else {
                // An expert too large for the VRAM tier belongs in the RAM tier.
                const auto ram_evicted = ram.insert(key, record.size);
                report.evictions += ram_evicted.size();
            }
        } else {
            const auto evicted = ram.insert(key, record.size);
            report.evictions += evicted.size();
        }
    }
    report.unique_experts = unique.size();
    finalize_report(report, per_token);
    return report;
}

TraceSimulationReport simulate_adaptive(const std::vector<TraceRecord>& records,
                                        const TraceSimulationConfig& config) {
    TraceSimulationReport report;
    report.policy = CachePolicy::adaptive;
    report.records = records.size();
    LfuCache vram(config.vram_cache_bytes);
    LfuCache ram(config.ram_cache_bytes);
    GhostHistory admission(admission_ghost_limit(records, config));
    std::unordered_set<ExpertKey> unique;
    std::map<std::uint64_t, ByteCount> per_token;
    for (const auto& record : records) {
        const ExpertKey key = make_key(record.layer, record.expert);
        unique.insert(key);
        per_token.try_emplace(record.token, 0U);
        if (vram.contains(key)) {
            ++report.vram_hits;
            vram.touch(key);
            continue;
        }
        if (ram.contains(key)) {
            ++report.ram_hits;
            ram.erase(key);
            if (config.vram_cache_bytes != 0U) {
                const auto evicted = vram.insert(key, record.size);
                if (vram.contains(key)) {
                    report.pcie_bytes += record.size;
                    report.evictions += evicted.size();
                    for (const auto& [victim, size] : evicted) {
                        const auto ram_evicted = ram.insert(victim, size);
                        report.evictions += ram_evicted.size();
                    }
                } else {
                    // An expert too large for the VRAM tier stays resident in the RAM tier.
                    ram.insert(key, record.size);
                }
            } else {
                ram.insert(key, record.size);
            }
            continue;
        }
        ++report.storage_misses;
        report.nvme_bytes += record.size;
        per_token[record.token] += record.size;
        // A first-seen expert is streamed for this access only; retention needs a second miss,
        // so one-off scans cannot displace experts that are actually reused.
        if (!admission.should_admit(key)) continue;
        if (config.vram_cache_bytes != 0U) {
            const auto evicted = vram.insert(key, record.size);
            if (vram.contains(key)) {
                report.pcie_bytes += record.size;
                report.evictions += evicted.size();
                for (const auto& [victim, size] : evicted) {
                    const auto ram_evicted = ram.insert(victim, size);
                    report.evictions += ram_evicted.size();
                }
            } else {
                const auto ram_evicted = ram.insert(key, record.size);
                report.evictions += ram_evicted.size();
            }
        } else {
            const auto ram_evicted = ram.insert(key, record.size);
            report.evictions += ram_evicted.size();
        }
    }
    report.unique_experts = unique.size();
    finalize_report(report, per_token);
    return report;
}

TraceSimulationReport simulate_no_cache(const std::vector<TraceRecord>& records) {
    TraceSimulationReport report;
    report.policy = CachePolicy::none;
    report.records = records.size();
    std::unordered_set<ExpertKey> unique;
    std::map<std::uint64_t, ByteCount> per_token;
    for (const auto& record : records) {
        unique.insert(make_key(record.layer, record.expert));
        ++report.storage_misses;
        report.nvme_bytes += record.size;
        per_token[record.token] += record.size;
    }
    report.unique_experts = unique.size();
    finalize_report(report, per_token);
    return report;
}

TraceSimulationReport simulate_static_hot(const std::vector<TraceRecord>& records, const TraceSimulationConfig& config) {
    struct Frequency {
        ExpertKey key;
        ByteCount size;
        std::size_t count;
    };
    std::unordered_map<ExpertKey, Frequency> frequencies;
    for (const auto& record : records) {
        const ExpertKey key = make_key(record.layer, record.expert);
        auto [iterator, inserted] = frequencies.emplace(key, Frequency{key, record.size, 0});
        ++iterator->second.count;
        if (inserted) iterator->second.size = record.size;
    }
    std::vector<Frequency> ranking;
    ranking.reserve(frequencies.size());
    for (const auto& [key, frequency] : frequencies) {
        (void)key;
        ranking.push_back(frequency);
    }
    std::sort(ranking.begin(), ranking.end(), [](const Frequency& left, const Frequency& right) {
        if (left.count != right.count) return left.count > right.count;
        if (left.size != right.size) return left.size < right.size;
        return left.key < right.key;
    });

    std::unordered_set<ExpertKey> vram_pins;
    std::unordered_set<ExpertKey> ram_pins;
    ByteCount vram_used = 0;
    ByteCount ram_used = 0;
    TraceSimulationReport report;
    report.policy = CachePolicy::static_hot;
    report.records = records.size();
    report.unique_experts = frequencies.size();

    for (const auto& item : ranking) {
        if (item.size <= config.vram_cache_bytes - std::min(vram_used, config.vram_cache_bytes)) {
            vram_pins.insert(item.key);
            vram_used += item.size;
            report.warmup_nvme_bytes += item.size;
        } else if (item.size <= config.ram_cache_bytes - std::min(ram_used, config.ram_cache_bytes)) {
            ram_pins.insert(item.key);
            ram_used += item.size;
            report.warmup_nvme_bytes += item.size;
        }
    }

    std::map<std::uint64_t, ByteCount> per_token;
    for (const auto& record : records) {
        const ExpertKey key = make_key(record.layer, record.expert);
        per_token.try_emplace(record.token, 0);
        if (vram_pins.contains(key)) {
            ++report.vram_hits;
        } else if (ram_pins.contains(key)) {
            ++report.ram_hits;
            if (config.vram_cache_bytes != 0) report.pcie_bytes += record.size;
        } else {
            ++report.storage_misses;
            report.nvme_bytes += record.size;
            per_token[record.token] += record.size;
            if (config.vram_cache_bytes != 0) report.pcie_bytes += record.size;
        }
    }
    finalize_report(report, per_token);
    return report;
}

}  // namespace

Result<CachePolicy> parse_cache_policy(std::string_view value) {
    const auto normalized = util::to_lower(util::trim(value));
    if (normalized == "none" || normalized == "no-cache") return CachePolicy::none;
    if (normalized == "lru") return CachePolicy::lru;
    if (normalized == "lfu") return CachePolicy::lfu;
    if (normalized == "static-hot" || normalized == "static_hot") return CachePolicy::static_hot;
    if (normalized == "adaptive") return CachePolicy::adaptive;
    return make_error(ErrorCode::invalid_argument, "Unknown trace cache policy", normalized);
}

std::string_view to_string(CachePolicy policy) noexcept {
    switch (policy) {
        case CachePolicy::none: return "no-cache";
        case CachePolicy::lru: return "lru";
        case CachePolicy::lfu: return "lfu";
        case CachePolicy::static_hot: return "static-hot";
        case CachePolicy::adaptive: return "adaptive";
    }
    return "lru";
}

Result<TraceSimulationReport> simulate_trace(const std::filesystem::path& trace_path,
                                             const TraceSimulationConfig& config) {
    if (config.policy != CachePolicy::none && config.vram_cache_bytes == 0 && config.ram_cache_bytes == 0) {
        return make_error(ErrorCode::invalid_argument, "At least one cache tier must have non-zero capacity");
    }
    auto records = load_trace(trace_path, config.max_records);
    if (!records) return records.error();
    try {
        TraceSimulationReport report;
        if (config.policy == CachePolicy::none) report = simulate_no_cache(records.value());
        else if (config.policy == CachePolicy::static_hot) report = simulate_static_hot(records.value(), config);
        else if (config.policy == CachePolicy::lfu) report = simulate_lfu(records.value(), config);
        else if (config.policy == CachePolicy::adaptive) report = simulate_adaptive(records.value(), config);
        else report = simulate_lru(records.value(), config);
        populate_reuse_metrics(records.value(), report);
        auto readiness = populate_ready_order_metrics(records.value(), report);
        if (!readiness) return readiness.error();
        return report;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::resource_exhausted, "Unable to allocate bounded trace simulation state");
    }
}

std::string to_text(const TraceSimulationReport& report) {
    std::ostringstream output;
    output << "Expert trace simulation\n"
           << "Policy:            " << to_string(report.policy) << '\n'
           << "Records:           " << report.records << '\n'
           << "Tokens:            " << report.tokens << '\n'
           << "Unique experts:    " << report.unique_experts << '\n'
           << "Cold accesses:     " << report.cold_expert_accesses << '\n'
           << "Reused accesses:   " << report.reused_expert_accesses << " ("
           << report.reuse_access_ratio * 100.0 << "%)\n"
           << "Mean reuse dist.:  " << report.average_unique_reuse_distance << " unique experts\n"
           << "P50 reuse dist.:   " << report.p50_unique_reuse_distance << " unique experts\n"
           << "P95 reuse dist.:   " << report.p95_unique_reuse_distance << " unique experts\n"
           << "Max reuse dist.:   " << report.maximum_unique_reuse_distance << " unique experts\n"
           << "VRAM hits:         " << report.vram_hits << " (" << report.vram_hit_ratio * 100.0 << "%)\n"
           << "RAM hits:          " << report.ram_hits << " (" << report.ram_hit_ratio * 100.0 << "%)\n"
           << "Storage misses:    " << report.storage_misses << " (" << report.storage_miss_ratio * 100.0 << "%)\n"
           << "NVMe traffic:      " << format_bytes(report.nvme_bytes) << '\n'
           << "PCIe traffic:      " << format_bytes(report.pcie_bytes) << '\n'
           << "Cache warmup:      " << format_bytes(report.warmup_nvme_bytes) << '\n'
           << "Average NVMe/token:" << ' ' << format_bytes(static_cast<ByteCount>(report.average_nvme_bytes_per_token)) << '\n'
           << "P50 NVMe/token:    " << format_bytes(report.p50_nvme_bytes_per_token) << '\n'
           << "P95 NVMe/token:    " << format_bytes(report.p95_nvme_bytes_per_token) << '\n'
           << "Max NVMe/token:    " << format_bytes(report.maximum_nvme_bytes_per_token) << '\n'
           << "Evictions:         " << report.evictions << '\n'
           << "Ready replay:      " << (report.ready_order_available ? "available" : "unavailable") << '\n';
    if (report.ready_order_available) {
        output << "Ready groups:      " << report.ready_order_groups << '\n'
               << "Fixed makespan:    " << report.fixed_order_makespan_ms << " ms\n"
               << "Ready makespan:    " << report.ready_order_makespan_ms << " ms\n"
               << "Fixed I/O wait:    " << report.fixed_order_wait_ms << " ms\n"
               << "Ready I/O wait:    " << report.ready_order_wait_ms << " ms\n"
               << "Ready saved:       " << report.ready_order_saved_ms << " ms ("
               << report.ready_order_savings_ratio * 100.0 << "%)\n"
               << "Compute reorders:  " << report.ready_order_reordered_experts << '\n';
    }
    return output.str();
}

std::string to_json(const TraceSimulationReport& report) {
    std::ostringstream output;
    output << '{'
           << "\"policy\":\"" << to_string(report.policy) << "\"," 
           << "\"records\":" << report.records << ','
           << "\"tokens\":" << report.tokens << ','
           << "\"unique_experts\":" << report.unique_experts << ','
           << "\"cold_expert_accesses\":" << report.cold_expert_accesses << ','
           << "\"reused_expert_accesses\":" << report.reused_expert_accesses << ','
           << "\"reuse_access_ratio\":" << report.reuse_access_ratio << ','
           << "\"average_unique_reuse_distance\":" << report.average_unique_reuse_distance << ','
           << "\"p50_unique_reuse_distance\":" << report.p50_unique_reuse_distance << ','
           << "\"p95_unique_reuse_distance\":" << report.p95_unique_reuse_distance << ','
           << "\"maximum_unique_reuse_distance\":" << report.maximum_unique_reuse_distance << ','
           << "\"vram_hits\":" << report.vram_hits << ','
           << "\"ram_hits\":" << report.ram_hits << ','
           << "\"storage_misses\":" << report.storage_misses << ','
           << "\"evictions\":" << report.evictions << ','
           << "\"nvme_bytes\":" << report.nvme_bytes << ','
           << "\"pcie_bytes\":" << report.pcie_bytes << ','
           << "\"warmup_nvme_bytes\":" << report.warmup_nvme_bytes << ','
           << "\"vram_hit_ratio\":" << report.vram_hit_ratio << ','
           << "\"ram_hit_ratio\":" << report.ram_hit_ratio << ','
           << "\"storage_miss_ratio\":" << report.storage_miss_ratio << ','
           << "\"average_nvme_bytes_per_token\":" << report.average_nvme_bytes_per_token << ','
           << "\"p50_nvme_bytes_per_token\":" << report.p50_nvme_bytes_per_token << ','
           << "\"p95_nvme_bytes_per_token\":" << report.p95_nvme_bytes_per_token << ','
           << "\"maximum_nvme_bytes_per_token\":" << report.maximum_nvme_bytes_per_token << ','
           << "\"ready_order_available\":" << (report.ready_order_available ? "true" : "false") << ','
           << "\"ready_order_groups\":" << report.ready_order_groups << ','
           << "\"ready_order_reordered_experts\":" << report.ready_order_reordered_experts << ','
           << "\"fixed_order_makespan_ms\":" << report.fixed_order_makespan_ms << ','
           << "\"ready_order_makespan_ms\":" << report.ready_order_makespan_ms << ','
           << "\"fixed_order_wait_ms\":" << report.fixed_order_wait_ms << ','
           << "\"ready_order_wait_ms\":" << report.ready_order_wait_ms << ','
           << "\"ready_order_saved_ms\":" << report.ready_order_saved_ms << ','
           << "\"ready_order_savings_ratio\":" << report.ready_order_savings_ratio
           << '}';
    return output.str();
}

}  // namespace vastloom::research

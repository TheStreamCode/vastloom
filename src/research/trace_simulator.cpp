#include "stargate/research/trace_simulator.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <list>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stargate/util/string.hpp"

namespace stargate::research {
namespace {

struct TraceRecord {
    std::uint64_t token{0};
    std::uint32_t layer{0};
    std::uint32_t expert{0};
    ByteCount size{0};
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
    std::unordered_map<ExpertKey, ByteCount> sizes;
    while (std::getline(input, line)) {
        ++line_number;
        const auto trimmed = util::trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        const auto fields = util::split(trimmed, ',', true);
        if (fields.size() != 4U && fields.size() != 9U) {
            return make_error(ErrorCode::invalid_format, "Trace rows require four baseline fields or nine telemetry fields", "line " + std::to_string(line_number));
        }
        if (util::to_lower(util::trim(fields[0])) == "token") {
            if (header_seen || !records.empty() || util::to_lower(util::trim(fields[1])) != "layer" ||
                util::to_lower(util::trim(fields[2])) != "expert" ||
                util::to_lower(util::trim(fields[3])) != "size_bytes") {
                return make_error(ErrorCode::invalid_format, "Invalid or repeated trace header", "line " + std::to_string(line_number));
            }
            header_seen = true;
            continue;
        }
        if (records.size() >= max_records) {
            return make_error(ErrorCode::resource_exhausted, "Trace exceeds configured record limit", path.string());
        }

        auto token = parse_u64(fields[0], "token"); if (!token) return token.error();
        auto layer = parse_u64(fields[1], "layer"); if (!layer || layer.value() > std::numeric_limits<std::uint32_t>::max()) return make_error(ErrorCode::invalid_format, "Layer is outside uint32 range", "line " + std::to_string(line_number));
        auto expert = parse_u64(fields[2], "expert"); if (!expert || expert.value() > std::numeric_limits<std::uint32_t>::max()) return make_error(ErrorCode::invalid_format, "Expert is outside uint32 range", "line " + std::to_string(line_number));
        auto record_size = parse_bytes(fields[3]); if (!record_size || record_size.value() == 0) return make_error(ErrorCode::invalid_format, "Expert size must be greater than zero", "line " + std::to_string(line_number));
        if (fields.size() == 9U) {
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
        }

        TraceRecord record{token.take_value(), static_cast<std::uint32_t>(layer.take_value()),
                           static_cast<std::uint32_t>(expert.take_value()), record_size.take_value()};
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
        ++iterator->second.frequency;
        iterator->second.last_use = ++clock_;
    }

    bool erase(ExpertKey key) {
        const auto iterator = entries_.find(key);
        if (iterator == entries_.end()) return false;
        used_ -= iterator->second.size;
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
        while (!entries_.empty() && used_ > capacity_ - size) {
            auto victim = entries_.begin();
            for (auto iterator = std::next(entries_.begin()); iterator != entries_.end(); ++iterator) {
                if (std::pair{iterator->second.frequency, iterator->second.last_use} <
                    std::pair{victim->second.frequency, victim->second.last_use}) {
                    victim = iterator;
                }
            }
            evicted.emplace_back(victim->first, victim->second.size);
            used_ -= victim->second.size;
            entries_.erase(victim);
        }
        entries_.emplace(key, Entry{size, 1U, ++clock_});
        used_ += size;
        return evicted;
    }

private:
    struct Entry {
        ByteCount size{0U};
        std::uint64_t frequency{0U};
        std::uint64_t last_use{0U};
    };
    ByteCount capacity_{0U};
    ByteCount used_{0U};
    std::uint64_t clock_{0U};
    std::unordered_map<ExpertKey, Entry> entries_;
};

ByteCount percentile_sorted(const std::vector<ByteCount>& values, double quantile) {
    if (values.empty()) return 0;
    const auto index = static_cast<std::size_t>(std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1U)];
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
                report.pcie_bytes += record.size;
                const auto evicted = vram.insert(key, record.size);
                report.evictions += evicted.size();
                for (const auto& [victim, size] : evicted) {
                    const auto ram_evicted = ram.insert(victim, size);
                    report.evictions += ram_evicted.size();
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
            report.pcie_bytes += record.size;
            const auto evicted = vram.insert(key, record.size);
            report.evictions += evicted.size();
            for (const auto& [victim, size] : evicted) {
                const auto ram_evicted = ram.insert(victim, size);
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
                report.pcie_bytes += record.size;
                const auto evicted = vram.insert(key, record.size);
                report.evictions += evicted.size();
                for (const auto& [victim, size] : evicted) {
                    const auto ram_evicted = ram.insert(victim, size);
                    report.evictions += ram_evicted.size();
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
            report.pcie_bytes += record.size;
            const auto evicted = vram.insert(key, record.size);
            report.evictions += evicted.size();
            for (const auto& [victim, size] : evicted) {
                const auto ram_evicted = ram.insert(victim, size);
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
    return make_error(ErrorCode::invalid_argument, "Unknown trace cache policy", normalized);
}

std::string_view to_string(CachePolicy policy) noexcept {
    switch (policy) {
        case CachePolicy::none: return "no-cache";
        case CachePolicy::lru: return "lru";
        case CachePolicy::lfu: return "lfu";
        case CachePolicy::static_hot: return "static-hot";
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
    if (config.policy == CachePolicy::none) return simulate_no_cache(records.value());
    if (config.policy == CachePolicy::static_hot) return simulate_static_hot(records.value(), config);
    if (config.policy == CachePolicy::lfu) return simulate_lfu(records.value(), config);
    return simulate_lru(records.value(), config);
}

std::string to_text(const TraceSimulationReport& report) {
    std::ostringstream output;
    output << "Expert trace simulation\n"
           << "Policy:            " << to_string(report.policy) << '\n'
           << "Records:           " << report.records << '\n'
           << "Tokens:            " << report.tokens << '\n'
           << "Unique experts:    " << report.unique_experts << '\n'
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
           << "Evictions:         " << report.evictions << '\n';
    return output.str();
}

std::string to_json(const TraceSimulationReport& report) {
    std::ostringstream output;
    output << '{'
           << "\"policy\":\"" << to_string(report.policy) << "\"," 
           << "\"records\":" << report.records << ','
           << "\"tokens\":" << report.tokens << ','
           << "\"unique_experts\":" << report.unique_experts << ','
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
           << "\"maximum_nvme_bytes_per_token\":" << report.maximum_nvme_bytes_per_token
           << '}';
    return output.str();
}

}  // namespace stargate::research

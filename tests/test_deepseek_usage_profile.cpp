#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "vastloom/inference/deepseek_v4_usage_profile.hpp"

namespace {

int failures = 0;

#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ << ": " #condition    \
                      << '\n';                                                                     \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

std::filesystem::path make_work_dir() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto base = std::filesystem::current_path() / "vastloom-usage-profile-test-tmp";
    std::filesystem::create_directories(base);
    const auto path = base / std::to_string(stamp);
    std::filesystem::create_directories(path);
    return path;
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << text;
}

void test_roundtrip_and_top() {
    const auto work = make_work_dir();
    const auto path = work / "profile.json";
    auto empty = vastloom::inference::load_deepseek_v4_usage_profile(
        path, "test/model", "rev1");
    CHECK(static_cast<bool>(empty));
    if (empty) {
        CHECK(empty.value().observations == 0U);
        std::vector<std::uint32_t> layer0{7U, 9U};
        std::vector<std::uint32_t> layer1{3U};
        CHECK(static_cast<bool>(vastloom::inference::record_deepseek_v4_usage(
            empty.value(), 0U, layer0)));
        CHECK(static_cast<bool>(vastloom::inference::record_deepseek_v4_usage(
            empty.value(), 1U, layer1)));
        CHECK(static_cast<bool>(vastloom::inference::record_deepseek_v4_usage(
            empty.value(), 0U, layer0)));
        CHECK(static_cast<bool>(vastloom::inference::save_deepseek_v4_usage_profile(
            path, empty.value())));
    }
    auto loaded = vastloom::inference::load_deepseek_v4_usage_profile(
        path, "test/model", "rev1");
    CHECK(static_cast<bool>(loaded));
    if (loaded) {
        CHECK(loaded.value().observations == 3U);
        CHECK(loaded.value().counts.at(0U).at(7U) == 2U);
        CHECK(loaded.value().counts.at(1U).at(3U) == 1U);
        const auto top = vastloom::inference::top_deepseek_v4_usage(
            loaded.value(), 0U, 1U);
        CHECK(top.size() == 1U);
        CHECK(top.empty() || (top[0].first == 7U && top[0].second == 2U));
    }
    std::filesystem::remove_all(work);
}

void test_malformed_profiles_are_refused() {
    const auto work = make_work_dir();
    const auto path = work / "profile.json";
    const std::string valid =
        R"({"format":"vastloom.deepseek-v4-usage-profile.v1","model_id":"test/model","revision":"rev1","observations":1,"layers":{"0":{"7":1}}})";
    write_text(path, valid);
    CHECK(static_cast<bool>(vastloom::inference::load_deepseek_v4_usage_profile(
        path, "test/model", "rev1")));
    CHECK(!vastloom::inference::load_deepseek_v4_usage_profile(path, "other/model", "rev1"));

    write_text(path, R"({"format":"vastloom.deepseek-v4-usage-profile.v1","model_id":"test/model","revision":"rev1","observations":1})");
    CHECK(!vastloom::inference::load_deepseek_v4_usage_profile(path, "test/model", "rev1"));

    write_text(path, R"({"format":"vastloom.deepseek-v4-usage-profile.v1","model_id":"test/model","revision":"rev1","observations":1,"layers":{"0":{"7":0}}})");
    CHECK(!vastloom::inference::load_deepseek_v4_usage_profile(path, "test/model", "rev1"));

    write_text(path, "not json");
    CHECK(!vastloom::inference::load_deepseek_v4_usage_profile(path, "test/model", "rev1"));
    std::filesystem::remove_all(work);
}

}  // namespace

int main() {
    test_roundtrip_and_top();
    test_malformed_profiles_are_refused();
    std::error_code error;
    std::filesystem::remove_all(
        std::filesystem::current_path() / "vastloom-usage-profile-test-tmp", error);
    if (failures != 0) {
        std::cerr << failures << " usage-profile test(s) failed\n";
        return 1;
    }
    std::cout << "Usage-profile tests PASS\n";
    return 0;
}

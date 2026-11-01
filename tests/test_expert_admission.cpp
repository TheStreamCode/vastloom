#include <cstddef>
#include <cstdint>
#include <iostream>

#include "vastloom/inference/adaptive_admission.hpp"

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

void test_two_hit_admission() {
    vastloom::inference::AdaptiveExpertAdmission admission(2U);
    CHECK(admission.ghost_limit() == 16U);
    CHECK(!admission.should_admit_miss(7U));
    CHECK(admission.should_admit_miss(7U));
    CHECK(admission.ghost_size() == 1U);
}

void test_scan_does_not_displace_a_hot_key() {
    // Minimal stand-in for a one-slot resident cache fed only by admitted misses.
    vastloom::inference::AdaptiveExpertAdmission admission(1U);
    std::uint32_t resident = 0U;
    bool resident_valid = false;
    const auto access = [&](std::uint32_t key) {
        if (resident_valid && resident == key) {
            return true;
        }
        if (admission.should_admit_miss(key)) {
            resident = key;
            resident_valid = true;
        }
        return false;
    };
    CHECK(!access(42U));
    CHECK(!access(42U));
    CHECK(resident_valid && resident == 42U);
    for (std::uint32_t key = 1000U; key < 1200U; ++key) {
        CHECK(!access(key));
        CHECK(access(42U));
    }
}

void test_ghost_history_is_bounded_and_expires() {
    vastloom::inference::AdaptiveExpertAdmission admission(1U);
    for (std::uint32_t key = 0U; key < 64U; ++key) {
        CHECK(!admission.should_admit_miss(key));
        CHECK(admission.ghost_size() <= admission.ghost_limit());
    }
    // Key 0 fell out of the bounded FIFO, so it needs a fresh first miss before admission.
    CHECK(!admission.should_admit_miss(0U));
    CHECK(admission.should_admit_miss(0U));
}

void test_ghost_limit_is_bounded() {
    CHECK(vastloom::inference::AdaptiveExpertAdmission(0U).ghost_limit() == 16U);
    CHECK(vastloom::inference::AdaptiveExpertAdmission(1024U).ghost_limit() == 4096U);
    CHECK(vastloom::inference::AdaptiveExpertAdmission(5000U).ghost_limit() == 4096U);
}

}  // namespace

int main() {
    test_two_hit_admission();
    test_scan_does_not_displace_a_hot_key();
    test_ghost_history_is_bounded_and_expires();
    test_ghost_limit_is_bounded();
    if (failures != 0) {
        std::cerr << failures << " adaptive-admission check(s) failed\n";
        return 1;
    }
    std::cout << "Adaptive expert admission tests passed\n";
    return 0;
}

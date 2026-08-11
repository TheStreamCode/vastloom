#include "vastloom/cuda/cuda_driver.hpp"

#include <array>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "vastloom/util/string.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define VASTLOOM_CUDA_CALL __stdcall
#elif defined(__linux__)
#include <dlfcn.h>
#define VASTLOOM_CUDA_CALL
#else
#define VASTLOOM_CUDA_CALL
#endif

namespace vastloom::cuda {
namespace {

using CuResult = int;
using CuDevice = int;
using CuContext = void*;
using CuDevicePointer = std::uint64_t;

constexpr CuResult kCudaSuccess = 0;
constexpr int kAttributeMaximumThreadsPerBlock = 1;
constexpr int kAttributeMultiprocessorCount = 16;
constexpr int kAttributeUnifiedAddressing = 41;
constexpr int kAttributeComputeCapabilityMajor = 75;
constexpr int kAttributeComputeCapabilityMinor = 76;
constexpr std::size_t kMaximumDeviceNameBytes = 256U;
constexpr std::size_t kMaximumAllocationTagBytes = 1'024U;

void quarantine_backend(std::shared_ptr<CudaDriverBackend> backend) noexcept {
    // A failed external release may leave a live device allocation/context. Keep the driver
    // module loaded until process teardown; static storage is reclaimed normally at exit.
    static std::mutex quarantine_mutex;
    static std::vector<std::shared_ptr<CudaDriverBackend>> quarantined;
    try {
        std::scoped_lock lock(quarantine_mutex);
        quarantined.push_back(std::move(backend));
    } catch (...) {
        // Accounting remains charged even if retaining the module cannot allocate bookkeeping.
    }
}

using CuInit = CuResult(VASTLOOM_CUDA_CALL*)(unsigned int);
using CuDriverGetVersion = CuResult(VASTLOOM_CUDA_CALL*)(int*);
using CuDeviceGetCount = CuResult(VASTLOOM_CUDA_CALL*)(int*);
using CuDeviceGet = CuResult(VASTLOOM_CUDA_CALL*)(CuDevice*, int);
using CuDeviceGetName = CuResult(VASTLOOM_CUDA_CALL*)(char*, int, CuDevice);
using CuDeviceTotalMem = CuResult(VASTLOOM_CUDA_CALL*)(std::size_t*, CuDevice);
using CuDeviceGetAttribute = CuResult(VASTLOOM_CUDA_CALL*)(int*, int, CuDevice);
using CuDevicePrimaryCtxRetain = CuResult(VASTLOOM_CUDA_CALL*)(CuContext*, CuDevice);
using CuDevicePrimaryCtxRelease = CuResult(VASTLOOM_CUDA_CALL*)(CuDevice);
using CuCtxPushCurrent = CuResult(VASTLOOM_CUDA_CALL*)(CuContext);
using CuCtxPopCurrent = CuResult(VASTLOOM_CUDA_CALL*)(CuContext*);
using CuMemGetInfo = CuResult(VASTLOOM_CUDA_CALL*)(std::size_t*, std::size_t*);
using CuMemAlloc = CuResult(VASTLOOM_CUDA_CALL*)(CuDevicePointer*, std::size_t);
using CuMemFree = CuResult(VASTLOOM_CUDA_CALL*)(CuDevicePointer);
using CuGetErrorName = CuResult(VASTLOOM_CUDA_CALL*)(CuResult, const char**);
using CuGetErrorString = CuResult(VASTLOOM_CUDA_CALL*)(CuResult, const char**);

class DynamicCudaBackend final : public CudaDriverBackend {
public:
    DynamicCudaBackend(const DynamicCudaBackend&) = delete;
    DynamicCudaBackend& operator=(const DynamicCudaBackend&) = delete;

    ~DynamicCudaBackend() override {
#if defined(_WIN32)
        if (library_handle_ != nullptr) FreeLibrary(library_handle_);
#elif defined(__linux__)
        if (library_handle_ != nullptr) dlclose(library_handle_);
#endif
    }

    [[nodiscard]] static Result<std::shared_ptr<CudaDriverBackend>> create() {
#if defined(_WIN32)
        auto backend = std::shared_ptr<DynamicCudaBackend>(new DynamicCudaBackend());
        backend->library_handle_ = LoadLibraryExW(
            L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        backend->library_name_ = "nvcuda.dll";
        if (backend->library_handle_ == nullptr) {
            return make_error(ErrorCode::unsupported,
                              "NVIDIA CUDA driver library is unavailable", "nvcuda.dll");
        }
#elif defined(__linux__)
        auto backend = std::shared_ptr<DynamicCudaBackend>(new DynamicCudaBackend());
        backend->library_handle_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        backend->library_name_ = "libcuda.so.1";
        if (backend->library_handle_ == nullptr) {
            const char* detail = dlerror();
            return make_error(ErrorCode::unsupported,
                              "NVIDIA CUDA driver library is unavailable",
                              detail == nullptr ? "libcuda.so.1" : detail);
        }
#else
        return make_error(ErrorCode::unsupported,
                          "The native CUDA driver boundary supports Windows and Linux only");
#endif

#if defined(_WIN32) || defined(__linux__)
        auto loaded = backend->load_functions();
        if (!loaded) return loaded.error();
        std::shared_ptr<CudaDriverBackend> result = std::move(backend);
        return result;
#endif
    }

    [[nodiscard]] Result<CudaDriverProbe> probe() override {
        std::scoped_lock lock(mutex_);
        auto initialized = initialize_locked();
        if (!initialized) return initialized.error();

        int driver_version = 0;
        CuResult status = driver_get_version_(&driver_version);
        if (status != kCudaSuccess) return cuda_error("cuDriverGetVersion", status);
        int device_count = 0;
        status = device_get_count_(&device_count);
        if (status != kCudaSuccess) return cuda_error("cuDeviceGetCount", status);
        if (device_count < 0 || device_count > 256) {
            return make_error(ErrorCode::invalid_format,
                              "CUDA driver returned an invalid device count",
                              std::to_string(device_count));
        }

        CudaDriverProbe result;
        result.library = library_name_;
        result.driver_version = driver_version;
        result.devices.reserve(static_cast<std::size_t>(device_count));
        for (int ordinal = 0; ordinal < device_count; ++ordinal) {
            auto device = get_device_locked(ordinal);
            if (!device) return device.error();
            CudaDeviceCapabilities capabilities;
            capabilities.ordinal = ordinal;
            std::array<char, kMaximumDeviceNameBytes> name{};
            status = device_get_name_(name.data(), static_cast<int>(name.size()), device.value());
            if (status != kCudaSuccess) return cuda_error("cuDeviceGetName", status);
            name.back() = '\0';
            capabilities.name = name.data();
            if (capabilities.name.empty()) {
                return make_error(ErrorCode::invalid_format,
                                  "CUDA driver returned an empty device name",
                                  std::to_string(ordinal));
            }
            std::size_t total_bytes = 0U;
            status = device_total_memory_(&total_bytes, device.value());
            if (status != kCudaSuccess) return cuda_error("cuDeviceTotalMem_v2", status);
            if (total_bytes == 0U) {
                return make_error(ErrorCode::invalid_format,
                                  "CUDA driver returned a zero-byte device capacity",
                                  capabilities.name);
            }
            capabilities.total_vram_bytes = static_cast<ByteCount>(total_bytes);
            auto major = attribute_locked(device.value(), kAttributeComputeCapabilityMajor,
                                          "compute capability major");
            auto minor = attribute_locked(device.value(), kAttributeComputeCapabilityMinor,
                                          "compute capability minor");
            auto multiprocessors = attribute_locked(device.value(), kAttributeMultiprocessorCount,
                                                    "multiprocessor count");
            auto maximum_threads = attribute_locked(device.value(), kAttributeMaximumThreadsPerBlock,
                                                    "maximum threads per block");
            auto unified = attribute_locked(device.value(), kAttributeUnifiedAddressing,
                                            "unified addressing");
            if (!major) return major.error();
            if (!minor) return minor.error();
            if (!multiprocessors) return multiprocessors.error();
            if (!maximum_threads) return maximum_threads.error();
            if (!unified) return unified.error();
            if (major.value() < 1 || minor.value() < 0 || multiprocessors.value() < 1 ||
                maximum_threads.value() < 1 ||
                (unified.value() != 0 && unified.value() != 1)) {
                return make_error(ErrorCode::invalid_format,
                                  "CUDA driver returned invalid device capabilities",
                                  capabilities.name);
            }
            capabilities.compute_capability_major = major.value();
            capabilities.compute_capability_minor = minor.value();
            capabilities.multiprocessor_count = multiprocessors.value();
            capabilities.maximum_threads_per_block = maximum_threads.value();
            capabilities.unified_addressing = unified.value() == 1;
            result.devices.push_back(std::move(capabilities));
        }
        return result;
    }

    [[nodiscard]] Result<CudaBackendAllocation> allocate(
        int device_ordinal,
        ByteCount bytes,
        ByteCount minimum_free_vram_after) override {
        std::scoped_lock lock(mutex_);
        auto initialized = initialize_locked();
        if (!initialized) return initialized.error();
        if (bytes > static_cast<ByteCount>(std::numeric_limits<std::size_t>::max())) {
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA allocation exceeds this process address size");
        }
        auto device = get_device_locked(device_ordinal);
        if (!device) return device.error();

        CuContext context = nullptr;
        CuResult status = primary_context_retain_(&context, device.value());
        if (status != kCudaSuccess) return cuda_error("cuDevicePrimaryCtxRetain", status);
        if (context == nullptr) {
            static_cast<void>(primary_context_release_(device.value()));
            return make_error(ErrorCode::internal_error,
                              "CUDA primary context retain returned a null context");
        }
        status = context_push_(context);
        if (status != kCudaSuccess) {
            static_cast<void>(primary_context_release_(device.value()));
            return cuda_error("cuCtxPushCurrent_v2", status);
        }

        std::size_t free_bytes = 0U;
        std::size_t total_bytes = 0U;
        status = memory_info_(&free_bytes, &total_bytes);
        if (status != kCudaSuccess) {
            pop_and_release_locked(device.value());
            return cuda_error("cuMemGetInfo_v2", status);
        }
        const ByteCount free = static_cast<ByteCount>(free_bytes);
        if (bytes > free || minimum_free_vram_after > free - bytes) {
            pop_and_release_locked(device.value());
            return make_error(ErrorCode::resource_exhausted,
                              "CUDA allocation would consume the live VRAM reserve",
                              "requested=" + std::to_string(bytes) +
                                  ", free=" + std::to_string(free) +
                                  ", reserve=" + std::to_string(minimum_free_vram_after));
        }

        CuDevicePointer pointer = 0U;
        status = memory_allocate_(&pointer, static_cast<std::size_t>(bytes));
        if (status != kCudaSuccess) {
            pop_and_release_locked(device.value());
            return cuda_error("cuMemAlloc_v2", status);
        }
        CuContext popped = nullptr;
        status = context_pop_(&popped);
        if (status != kCudaSuccess) {
            static_cast<void>(memory_free_(pointer));
            static_cast<void>(primary_context_release_(device.value()));
            return cuda_error("cuCtxPopCurrent_v2", status);
        }
        if (popped != context) {
            static_cast<void>(context_push_(context));
            static_cast<void>(memory_free_(pointer));
            static_cast<void>(context_pop_(&popped));
            static_cast<void>(primary_context_release_(device.value()));
            return make_error(ErrorCode::internal_error,
                              "CUDA context stack returned a different context");
        }
        return CudaBackendAllocation{reinterpret_cast<std::uintptr_t>(context), pointer,
                                     bytes, device_ordinal};
    }

    [[nodiscard]] Result<void> release(const CudaBackendAllocation& allocation) override {
        std::scoped_lock lock(mutex_);
        auto initialized = initialize_locked();
        if (!initialized) return initialized.error();
        if (allocation.context == 0U || allocation.device_pointer == 0U ||
            allocation.bytes == 0U || allocation.device_ordinal < 0) {
            return make_error(ErrorCode::invalid_argument,
                              "Invalid CUDA allocation handle supplied for release");
        }
        auto device = get_device_locked(allocation.device_ordinal);
        if (!device) return device.error();
        auto* context = reinterpret_cast<CuContext>(allocation.context);
        CuResult status = context_push_(context);
        if (status != kCudaSuccess) return cuda_error("cuCtxPushCurrent_v2", status);
        status = memory_free_(allocation.device_pointer);
        if (status != kCudaSuccess) {
            CuContext popped = nullptr;
            static_cast<void>(context_pop_(&popped));
            return cuda_error("cuMemFree_v2", status);
        }

        // The device allocation is gone at this point. Context-stack/refcount cleanup is best
        // effort because retrying cuMemFree after either of these calls fails would double-free.
        CuContext popped = nullptr;
        static_cast<void>(context_pop_(&popped));
        static_cast<void>(primary_context_release_(device.value()));
        return {};
    }

private:
    DynamicCudaBackend() = default;

    template <typename Function>
    [[nodiscard]] Result<void> load_symbol(Function& target,
                                           std::initializer_list<const char*> names) {
        for (const char* name : names) {
#if defined(_WIN32)
            const FARPROC raw = GetProcAddress(library_handle_, name);
#elif defined(__linux__)
            void* const raw = dlsym(library_handle_, name);
#endif
            if (raw != nullptr) {
                static_assert(sizeof(target) == sizeof(raw));
                std::memcpy(&target, &raw, sizeof(target));
                return {};
            }
        }
        return make_error(ErrorCode::unsupported,
                          "CUDA driver is missing a required entry point",
                          names.size() == 0U ? "unknown" : *names.begin());
    }

    [[nodiscard]] Result<void> load_functions() {
#define VASTLOOM_LOAD_CUDA(member, type, ...)                  \
    do {                                                       \
        auto loaded = load_symbol<type>(member, {__VA_ARGS__}); \
        if (!loaded) return loaded.error();                    \
    } while (false)
        VASTLOOM_LOAD_CUDA(init_, CuInit, "cuInit");
        VASTLOOM_LOAD_CUDA(driver_get_version_, CuDriverGetVersion, "cuDriverGetVersion");
        VASTLOOM_LOAD_CUDA(device_get_count_, CuDeviceGetCount, "cuDeviceGetCount");
        VASTLOOM_LOAD_CUDA(device_get_, CuDeviceGet, "cuDeviceGet");
        VASTLOOM_LOAD_CUDA(device_get_name_, CuDeviceGetName, "cuDeviceGetName");
        VASTLOOM_LOAD_CUDA(device_total_memory_, CuDeviceTotalMem, "cuDeviceTotalMem_v2");
        VASTLOOM_LOAD_CUDA(device_get_attribute_, CuDeviceGetAttribute,
                           "cuDeviceGetAttribute");
        VASTLOOM_LOAD_CUDA(primary_context_retain_, CuDevicePrimaryCtxRetain,
                           "cuDevicePrimaryCtxRetain");
        VASTLOOM_LOAD_CUDA(primary_context_release_, CuDevicePrimaryCtxRelease,
                           "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease");
        VASTLOOM_LOAD_CUDA(context_push_, CuCtxPushCurrent,
                           "cuCtxPushCurrent_v2", "cuCtxPushCurrent");
        VASTLOOM_LOAD_CUDA(context_pop_, CuCtxPopCurrent,
                           "cuCtxPopCurrent_v2", "cuCtxPopCurrent");
        VASTLOOM_LOAD_CUDA(memory_info_, CuMemGetInfo, "cuMemGetInfo_v2");
        VASTLOOM_LOAD_CUDA(memory_allocate_, CuMemAlloc, "cuMemAlloc_v2");
        VASTLOOM_LOAD_CUDA(memory_free_, CuMemFree, "cuMemFree_v2");
        VASTLOOM_LOAD_CUDA(error_name_, CuGetErrorName, "cuGetErrorName");
        VASTLOOM_LOAD_CUDA(error_string_, CuGetErrorString, "cuGetErrorString");
#undef VASTLOOM_LOAD_CUDA
        return {};
    }

    [[nodiscard]] Result<void> initialize_locked() {
        if (initialized_) return {};
        const CuResult status = init_(0U);
        if (status != kCudaSuccess) return cuda_error("cuInit", status);
        initialized_ = true;
        return {};
    }

    [[nodiscard]] Result<CuDevice> get_device_locked(int ordinal) const {
        if (ordinal < 0) {
            return make_error(ErrorCode::invalid_argument,
                              "CUDA device ordinal cannot be negative");
        }
        int count = 0;
        CuResult status = device_get_count_(&count);
        if (status != kCudaSuccess) return cuda_error("cuDeviceGetCount", status);
        if (ordinal >= count) {
            return make_error(ErrorCode::not_found, "CUDA device ordinal is unavailable",
                              std::to_string(ordinal));
        }
        CuDevice device = 0;
        status = device_get_(&device, ordinal);
        if (status != kCudaSuccess) return cuda_error("cuDeviceGet", status);
        return device;
    }

    [[nodiscard]] Result<int> attribute_locked(CuDevice device, int attribute,
                                               std::string_view label) const {
        int value = 0;
        const CuResult status = device_get_attribute_(&value, attribute, device);
        if (status != kCudaSuccess) {
            auto error = cuda_error("cuDeviceGetAttribute", status);
            error.context = std::string(label) + ": " + error.context;
            return error;
        }
        return value;
    }

    [[nodiscard]] Error cuda_error(std::string_view operation, CuResult status) const {
        const char* name = nullptr;
        const char* description = nullptr;
        static_cast<void>(error_name_(status, &name));
        static_cast<void>(error_string_(status, &description));
        std::string context = "status=" + std::to_string(status);
        if (name != nullptr) context += ", " + std::string(name);
        if (description != nullptr) context += ": " + std::string(description);
        return make_error(ErrorCode::io_error,
                          "CUDA driver call failed: " + std::string(operation),
                          std::move(context));
    }

    void pop_and_release_locked(CuDevice device) const noexcept {
        CuContext popped = nullptr;
        static_cast<void>(context_pop_(&popped));
        static_cast<void>(primary_context_release_(device));
    }

#if defined(_WIN32)
    HMODULE library_handle_{nullptr};
#elif defined(__linux__)
    void* library_handle_{nullptr};
#endif
    std::string library_name_;
    mutable std::mutex mutex_;
    bool initialized_{false};
    CuInit init_{nullptr};
    CuDriverGetVersion driver_get_version_{nullptr};
    CuDeviceGetCount device_get_count_{nullptr};
    CuDeviceGet device_get_{nullptr};
    CuDeviceGetName device_get_name_{nullptr};
    CuDeviceTotalMem device_total_memory_{nullptr};
    CuDeviceGetAttribute device_get_attribute_{nullptr};
    CuDevicePrimaryCtxRetain primary_context_retain_{nullptr};
    CuDevicePrimaryCtxRelease primary_context_release_{nullptr};
    CuCtxPushCurrent context_push_{nullptr};
    CuCtxPopCurrent context_pop_{nullptr};
    CuMemGetInfo memory_info_{nullptr};
    CuMemAlloc memory_allocate_{nullptr};
    CuMemFree memory_free_{nullptr};
    CuGetErrorName error_name_{nullptr};
    CuGetErrorString error_string_{nullptr};
};

}  // namespace

CudaAllocation::CudaAllocation(std::shared_ptr<CudaDriverBackend> backend,
                               CudaBackendAllocation allocation,
                               safety::Reservation reservation)
    : backend_(std::move(backend)),
      allocation_(allocation),
      reservation_(std::move(reservation)) {}

CudaAllocation::CudaAllocation(CudaAllocation&& other) noexcept
    : backend_(std::move(other.backend_)),
      allocation_(std::exchange(other.allocation_, {})),
      reservation_(std::move(other.reservation_)) {}

CudaAllocation& CudaAllocation::operator=(CudaAllocation&& other) noexcept {
    if (this != &other) {
        release_for_destruction();
        backend_ = std::move(other.backend_);
        allocation_ = std::exchange(other.allocation_, {});
        reservation_ = std::move(other.reservation_);
    }
    return *this;
}

CudaAllocation::~CudaAllocation() {
    release_for_destruction();
}

bool CudaAllocation::valid() const noexcept {
    return backend_ != nullptr && allocation_.context != 0U &&
           allocation_.device_pointer != 0U && allocation_.bytes != 0U &&
           reservation_.valid();
}

std::uint64_t CudaAllocation::device_pointer() const noexcept {
    return allocation_.device_pointer;
}

ByteCount CudaAllocation::bytes() const noexcept {
    return allocation_.bytes;
}

int CudaAllocation::device_ordinal() const noexcept {
    return allocation_.device_ordinal;
}

Result<void> CudaAllocation::release() {
    if (!valid()) return {};
    Result<void> released;
    try {
        released = backend_->release(allocation_);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw while releasing an allocation", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown release exception");
    }
    if (!released) return released.error();
    allocation_ = {};
    reservation_.release();
    backend_.reset();
    return {};
}

void CudaAllocation::release_for_destruction() noexcept {
    if (!valid()) return;
    try {
        auto released = release();
        if (released) return;
    } catch (...) {
    }
    // The device allocation may still be live. Preserve fail-closed Governor accounting rather
    // than allowing a second allocation to consume bytes that the backend failed to free.
    reservation_.abandon_accounting_on_external_release_failure();
    allocation_ = {};
    quarantine_backend(std::move(backend_));
}

CudaDriver::CudaDriver(std::shared_ptr<CudaDriverBackend> backend)
    : backend_(std::move(backend)) {}

Result<CudaDriver> CudaDriver::load() {
    auto backend = DynamicCudaBackend::create();
    if (!backend) return backend.error();
    return CudaDriver(backend.take_value());
}

Result<CudaDriver> CudaDriver::from_backend_for_testing(
    std::shared_ptr<CudaDriverBackend> backend) {
    if (backend == nullptr) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA test backend cannot be null");
    }
    return CudaDriver(std::move(backend));
}

Result<CudaDriverProbe> CudaDriver::probe() const {
    if (backend_ == nullptr) {
        return make_error(ErrorCode::internal_error, "CUDA driver backend is unavailable");
    }
    try {
        return backend_->probe();
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw during capability probing", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown probe exception");
    }
}

Result<CudaAllocation> CudaDriver::allocate(
    int device_ordinal,
    ByteCount bytes,
    ByteCount minimum_free_vram_after,
    safety::ResourceGovernor& governor,
    std::string tag) const {
    if (backend_ == nullptr) {
        return make_error(ErrorCode::internal_error, "CUDA driver backend is unavailable");
    }
    if (device_ordinal < 0) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA device ordinal cannot be negative");
    }
    if (bytes == 0U) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation size cannot be zero", std::move(tag));
    }
    if (tag.empty() || tag.size() > kMaximumAllocationTagBytes) {
        return make_error(ErrorCode::invalid_argument,
                          "CUDA allocation tag must contain 1 to 1024 bytes");
    }
    auto reservation = governor.try_reserve(safety::ResourceKind::vram, bytes, tag);
    if (!reservation) return reservation.error();

    Result<CudaBackendAllocation> allocated = make_error(
        ErrorCode::internal_error, "CUDA backend allocation was not attempted");
    try {
        allocated = backend_->allocate(device_ordinal, bytes, minimum_free_vram_after);
    } catch (const std::exception& exception) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw while allocating VRAM", exception.what());
    } catch (...) {
        return make_error(ErrorCode::internal_error,
                          "CUDA backend threw an unknown allocation exception");
    }
    if (!allocated) return allocated.error();
    const auto& handle = allocated.value();
    if (handle.context == 0U || handle.device_pointer == 0U || handle.bytes != bytes ||
        handle.device_ordinal != device_ordinal) {
        bool cleanup_succeeded = handle.context == 0U || handle.device_pointer == 0U;
        if (handle.context != 0U && handle.device_pointer != 0U) {
            try {
                cleanup_succeeded = static_cast<bool>(backend_->release(handle));
            } catch (...) {
                cleanup_succeeded = false;
            }
        }
        if (!cleanup_succeeded) {
            reservation.value().abandon_accounting_on_external_release_failure();
            quarantine_backend(backend_);
            return make_error(
                ErrorCode::integrity_error,
                "CUDA backend returned an invalid handle and cleanup failed; VRAM accounting remains charged");
        }
        return make_error(ErrorCode::integrity_error,
                          "CUDA backend returned an invalid allocation handle");
    }
    return CudaAllocation(backend_, allocated.take_value(), reservation.take_value());
}

std::string cuda_driver_version_text(int encoded_version) {
    if (encoded_version <= 0) return "unknown";
    const int major = encoded_version / 1'000;
    const int minor = (encoded_version % 1'000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

std::string to_json(const CudaDriverProbe& probe) {
    std::ostringstream output;
    output << "{\"ok\":true,\"library\":\"" << util::json_escape(probe.library)
           << "\",\"driver_version\":" << probe.driver_version
           << ",\"driver_version_text\":\""
           << util::json_escape(cuda_driver_version_text(probe.driver_version))
           << "\",\"devices\":[";
    for (std::size_t index = 0U; index < probe.devices.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& device = probe.devices[index];
        output << "{\"ordinal\":" << device.ordinal << ",\"name\":\""
               << util::json_escape(device.name) << "\",\"total_vram_bytes\":"
               << device.total_vram_bytes << ",\"compute_capability_major\":"
               << device.compute_capability_major << ",\"compute_capability_minor\":"
               << device.compute_capability_minor << ",\"multiprocessor_count\":"
               << device.multiprocessor_count << ",\"maximum_threads_per_block\":"
               << device.maximum_threads_per_block << ",\"unified_addressing\":"
               << (device.unified_addressing ? "true" : "false") << '}';
    }
    output << "]}";
    return output.str();
}

std::string to_text(const CudaDriverProbe& probe) {
    std::ostringstream output;
    output << "Vastloom CUDA driver report\n"
           << "Library:        " << probe.library << '\n'
           << "Driver API:     " << cuda_driver_version_text(probe.driver_version) << '\n'
           << "Devices:        " << probe.devices.size() << '\n';
    for (const auto& device : probe.devices) {
        output << "GPU " << device.ordinal << ":          " << device.name << '\n'
               << "  Capability:   " << device.compute_capability_major << '.'
               << device.compute_capability_minor << '\n'
               << "  Total VRAM:   " << format_bytes(device.total_vram_bytes) << '\n'
               << "  SMs/threads:  " << device.multiprocessor_count << '/'
               << device.maximum_threads_per_block << '\n'
               << "  UVA:          " << (device.unified_addressing ? "yes" : "no") << '\n';
    }
    return output.str();
}

}  // namespace vastloom::cuda

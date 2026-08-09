#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace vastloom {

enum class ErrorCode {
    invalid_argument,
    invalid_format,
    not_found,
    permission_denied,
    io_error,
    integrity_error,
    unsafe_path,
    unsupported,
    resource_exhausted,
    internal_error,
};

struct Error {
    ErrorCode code{ErrorCode::internal_error};
    std::string message;
    std::string context;
};

inline Error make_error(ErrorCode code, std::string message, std::string context = {}) {
    return Error{code, std::move(message), std::move(context)};
}

template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Error error) : data_(std::move(error)) {}

    [[nodiscard]] bool ok() const noexcept { return std::holds_alternative<T>(data_); }
    explicit operator bool() const noexcept { return ok(); }

    [[nodiscard]] T& value() {
        if (!ok()) {
            throw std::logic_error("Result does not contain a value");
        }
        return std::get<T>(data_);
    }

    [[nodiscard]] const T& value() const {
        if (!ok()) {
            throw std::logic_error("Result does not contain a value");
        }
        return std::get<T>(data_);
    }

    [[nodiscard]] T take_value() {
        if (!ok()) {
            throw std::logic_error("Result does not contain a value");
        }
        return std::move(std::get<T>(data_));
    }

    [[nodiscard]] Error& error() {
        if (ok()) {
            throw std::logic_error("Result does not contain an error");
        }
        return std::get<Error>(data_);
    }

    [[nodiscard]] const Error& error() const {
        if (ok()) {
            throw std::logic_error("Result does not contain an error");
        }
        return std::get<Error>(data_);
    }

private:
    std::variant<T, Error> data_;
};

template <>
class Result<void> {
public:
    Result() = default;
    Result(Error error) : error_(std::move(error)), ok_(false) {}

    [[nodiscard]] bool ok() const noexcept { return ok_; }
    explicit operator bool() const noexcept { return ok(); }

    [[nodiscard]] const Error& error() const {
        if (ok_) {
            throw std::logic_error("Result does not contain an error");
        }
        return error_;
    }

private:
    Error error_{};
    bool ok_{true};
};

}  // namespace vastloom

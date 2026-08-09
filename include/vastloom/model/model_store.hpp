#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

#include "vastloom/core/bytes.hpp"
#include "vastloom/core/status.hpp"

namespace vastloom::model {

struct ModelStoreOptions {
    ByteCount minimum_free_after_write{300U * GiB};
};

struct ModelStoreLayout {
    std::filesystem::path root;
    std::filesystem::path original;
    std::filesystem::path optimized;
    std::filesystem::path cache;
    std::filesystem::path scratch;
    std::filesystem::path state;
    std::filesystem::path telemetry;
    ByteCount minimum_free_after_write{0};
};

class ModelStore {
public:
    [[nodiscard]] static Result<ModelStore> initialize(const std::filesystem::path& root,
                                                       ModelStoreOptions options = {});
    [[nodiscard]] static Result<ModelStore> open(const std::filesystem::path& root,
                                                 ModelStoreOptions options = {});

    [[nodiscard]] const ModelStoreLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] Result<void> preflight_write(ByteCount bytes) const;
    [[nodiscard]] Result<std::filesystem::path> immutable_model_path(std::string_view model_id,
                                                                     std::string_view revision) const;

private:
    explicit ModelStore(ModelStoreLayout layout) : layout_(std::move(layout)) {}
    ModelStoreLayout layout_;
};

[[nodiscard]] std::string to_json(const ModelStoreLayout& layout);
[[nodiscard]] std::string to_text(const ModelStoreLayout& layout);

}  // namespace vastloom::model

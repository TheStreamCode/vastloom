#pragma once

#include <string>
#include <vector>

namespace vastloom::cli {

[[nodiscard]] int command_chat(const std::vector<std::string>& args);
[[nodiscard]] int command_serve(const std::vector<std::string>& args);
[[nodiscard]] int command_model_info(const std::vector<std::string>& args);
[[nodiscard]] int command_self_test(const std::vector<std::string>& args);

}  // namespace vastloom::cli

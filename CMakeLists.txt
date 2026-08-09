cmake_minimum_required(VERSION 3.24)

project(
    stargate_local_ai
    VERSION 0.3.0
    DESCRIPTION "Safety-first local inference runtime for oversized MoE models"
    LANGUAGES CXX
)

option(STARGATE_BUILD_TESTS "Build Stargate tests" ON)
option(STARGATE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(STARGATE_ENABLE_THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
option(STARGATE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
set(STARGATE_SMOLLM2_MODEL "" CACHE PATH "Optional external pinned SmolLM2 checkpoint for full reference tests")
set(STARGATE_OLMOE_MODEL "" CACHE PATH "Optional external pinned OLMoE checkpoint or metadata import for inspection tests")

if(STARGATE_ENABLE_SANITIZERS AND STARGATE_ENABLE_THREAD_SANITIZER)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer must use separate builds")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(StargateWarnings)

find_package(Threads REQUIRED)

add_library(stargate_core STATIC
    src/core/bytes.cpp
    src/hardware/hardware_probe.cpp
    src/inference/gpt2_tokenizer.cpp
    src/inference/llama_model.cpp
    src/inference/olmoe_model.cpp
    src/inference/tiny_model.cpp
    src/io/bounded_io_pool.cpp
    src/model/imported_checkpoint.cpp
    src/model/manifest.cpp
    src/model/model_store.cpp
    src/model/safetensors.cpp
    src/model/verifier.cpp
    src/planner/execution_planner.cpp
    src/research/trace_simulator.cpp
    src/safety/heartbeat.cpp
    src/safety/resource_governor.cpp
    src/safety/runtime_policy.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)

add_library(Stargate::Core ALIAS stargate_core)

target_include_directories(stargate_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(stargate_core PUBLIC cxx_std_20)
target_link_libraries(stargate_core PUBLIC Threads::Threads)
stargate_set_project_warnings(stargate_core)
stargate_enable_sanitizers(stargate_core)

if(WIN32)
    target_compile_definitions(stargate_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(stargate_core PRIVATE Psapi)
endif()

add_executable(stargate
    src/cli/main.cpp
    src/cli/tiny_commands.cpp
)
target_link_libraries(stargate PRIVATE Stargate::Core)
target_compile_features(stargate PRIVATE cxx_std_20)
stargate_set_project_warnings(stargate)
stargate_enable_sanitizers(stargate)

add_executable(stargate-watchdog src/watchdog/main.cpp)
target_link_libraries(stargate-watchdog PRIVATE Stargate::Core)
target_compile_features(stargate-watchdog PRIVATE cxx_std_20)
stargate_set_project_warnings(stargate-watchdog)
stargate_enable_sanitizers(stargate-watchdog)

include(CTest)
if(BUILD_TESTING AND STARGATE_BUILD_TESTS)
    add_executable(stargate_tests
        tests/test_main.cpp
    )
    target_link_libraries(stargate_tests PRIVATE Stargate::Core)
    target_compile_features(stargate_tests PRIVATE cxx_std_20)
    target_compile_definitions(stargate_tests PRIVATE STARGATE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    stargate_set_project_warnings(stargate_tests)
    stargate_enable_sanitizers(stargate_tests)
    add_test(NAME stargate_tests COMMAND stargate_tests)
    add_test(
        NAME stargate_cli_tiny_chat
        COMMAND stargate chat
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/models/stargate-tinychat-v1.sgm
            --prompt "Quanto fa 2+2?"
            --max-tokens 64
            --no-stream
            --no-stats
    )
    set_tests_properties(stargate_cli_tiny_chat PROPERTIES PASS_REGULAR_EXPRESSION "2[+]2 fa 4[.]")
    add_test(
        NAME stargate_cli_tiny_self_test
        COMMAND stargate self-test
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/models/stargate-tinychat-v1.sgm
    )
    set_tests_properties(stargate_cli_tiny_self_test PROPERTIES PASS_REGULAR_EXPRESSION "Result: PASS")

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME stargate_cli_tiny_integration
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tiny_chat_cli.py
                --stargate $<TARGET_FILE:stargate>
                --manifest ${CMAKE_CURRENT_SOURCE_DIR}/examples/models/stargate-tinychat-v1.sgm
                --vectors ${CMAKE_CURRENT_SOURCE_DIR}/examples/models/stargate-tinychat-v1.tests.json
        )
        set_tests_properties(stargate_cli_tiny_integration PROPERTIES TIMEOUT 60)
        add_test(
            NAME stargate_hf_importer_offline
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_import_hf_model.py
        )
        set_tests_properties(stargate_hf_importer_offline PROPERTIES TIMEOUT 30)
        if(STARGATE_SMOLLM2_MODEL)
            add_test(
                NAME stargate_smollm2_reference
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_smollm2_cli.py
                    --stargate $<TARGET_FILE:stargate>
                    --model ${STARGATE_SMOLLM2_MODEL}
                    --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/smollm2_reference.json
            )
            set_tests_properties(stargate_smollm2_reference PROPERTIES TIMEOUT 300)
        endif()
        if(STARGATE_OLMOE_MODEL)
            add_test(
                NAME stargate_olmoe_inspection
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_olmoe_cli.py
                    --stargate $<TARGET_FILE:stargate>
                    --model ${STARGATE_OLMOE_MODEL}
            )
            set_tests_properties(stargate_olmoe_inspection PROPERTIES TIMEOUT 60)
        endif()
    else()
        message(STATUS "Python3 not found; skipping TinyChat CLI integration test")
    endif()
endif()

install(TARGETS stargate stargate-watchdog stargate_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY examples/models/ DESTINATION share/stargate/models)

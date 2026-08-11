cmake_minimum_required(VERSION 3.24)

project(
    Vastloom
    VERSION 0.4.0
    DESCRIPTION "Run giant models locally without surrendering control of the host"
    HOMEPAGE_URL "https://github.com/TheStreamCode/vastloom"
    LANGUAGES CXX
)

function(vastloom_import_legacy_bool legacy_name current_name description default_value)
    if(DEFINED ${legacy_name} AND NOT DEFINED ${current_name})
        set(${current_name} "${${legacy_name}}" CACHE BOOL "${description}" FORCE)
        message(DEPRECATION "${legacy_name} was renamed to ${current_name}")
    elseif(NOT DEFINED ${current_name})
        set(${current_name} "${default_value}" CACHE BOOL "${description}")
    endif()
endfunction()

function(vastloom_import_legacy_path legacy_name current_name description)
    if(DEFINED ${legacy_name} AND NOT DEFINED ${current_name})
        set(${current_name} "${${legacy_name}}" CACHE PATH "${description}" FORCE)
        message(DEPRECATION "${legacy_name} was renamed to ${current_name}")
    elseif(NOT DEFINED ${current_name})
        set(${current_name} "" CACHE PATH "${description}")
    endif()
endfunction()

vastloom_import_legacy_bool(STARGATE_BUILD_TESTS VASTLOOM_BUILD_TESTS "Build Vastloom tests" ON)
vastloom_import_legacy_bool(STARGATE_ENABLE_SANITIZERS VASTLOOM_ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
vastloom_import_legacy_bool(STARGATE_ENABLE_THREAD_SANITIZER VASTLOOM_ENABLE_THREAD_SANITIZER
    "Enable ThreadSanitizer" OFF)
vastloom_import_legacy_bool(STARGATE_WARNINGS_AS_ERRORS VASTLOOM_WARNINGS_AS_ERRORS
    "Treat compiler warnings as errors" OFF)
vastloom_import_legacy_path(STARGATE_SMOLLM2_MODEL VASTLOOM_SMOLLM2_MODEL
    "Optional external pinned SmolLM2 checkpoint for full reference tests")
vastloom_import_legacy_path(STARGATE_OLMOE_MODEL VASTLOOM_OLMOE_MODEL
    "Optional external pinned OLMoE checkpoint or metadata import for inspection tests")
vastloom_import_legacy_path(STARGATE_OLMOE_FULL_MODEL VASTLOOM_OLMOE_FULL_MODEL
    "Optional complete pinned OLMoE checkpoint for streamed runtime tests")
set(VASTLOOM_DEEPSEEK_V4_MODEL "" CACHE PATH
    "Optional pinned DeepSeek V4 Flash metadata/header import for inspection tests")
set(VASTLOOM_DEEPSEEK_V4_GGUF_HEADER "" CACHE FILEPATH
    "Optional pinned DeepSeek V4 Flash 0731 GGUF split-zero metadata shard")
set(VASTLOOM_DEEPSEEK_V4_GGUF_CHECKPOINT "" CACHE PATH
    "Optional pinned DeepSeek V4 Flash 0731 UD-IQ1_S imported checkpoint")
set(VASTLOOM_DEEPSEEK_V4_IQ1S_SLICE "" CACHE FILEPATH
    "Optional pinned DeepSeek V4 Flash 0731 layer-0 expert-0 IQ1_S gate slice")
set(VASTLOOM_DEEPSEEK_V4_IQ1S_UP_SLICE "" CACHE FILEPATH
    "Optional pinned DeepSeek V4 Flash 0731 layer-0 expert-0 IQ1_S up slice")
set(VASTLOOM_DEEPSEEK_V4_IQ3XXS_SLICE "" CACHE FILEPATH
    "Optional pinned DeepSeek V4 Flash 0731 layer-0 expert-0 IQ3_XXS down slice")

option(VASTLOOM_BUILD_TESTS "Build Vastloom tests" ON)
option(VASTLOOM_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(VASTLOOM_ENABLE_THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
option(VASTLOOM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(VASTLOOM_ENABLE_SANITIZERS AND VASTLOOM_ENABLE_THREAD_SANITIZER)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer must use separate builds")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(VastloomWarnings)

find_package(Threads REQUIRED)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(vastloom_core STATIC
    src/core/bytes.cpp
    src/hardware/hardware_probe.cpp
    src/inference/gpt2_tokenizer.cpp
    src/inference/ggml_quantization.cpp
    src/inference/deepseek_v4_gguf.cpp
    src/inference/deepseek_v4_probe.cpp
    src/inference/deepseek_v4_range_artifact.cpp
    src/inference/deepseek_v4_model.cpp
    src/inference/deepseek_v4_graph.cpp
    src/inference/deepseek_v4_expert_provider.cpp
    src/inference/deepseek_v4_reference.cpp
    src/inference/llama_model.cpp
    src/inference/olmoe_model.cpp
    src/inference/olmoe_runtime.cpp
    src/inference/tiny_model.cpp
    src/io/bounded_io_pool.cpp
    src/io/device_io_counters.cpp
    src/io/process_io_counters.cpp
    src/model/bound_range_loader.cpp
    src/model/expert_loader.cpp
    src/model/gguf.cpp
    src/model/imported_checkpoint.cpp
    src/model/layer_streaming.cpp
    src/model/manifest.cpp
    src/model/model_store.cpp
    src/model/range_artifact.cpp
    src/model/residency.cpp
    src/model/safetensors.cpp
    src/model/tensor_catalog.cpp
    src/model/verifier.cpp
    src/planner/execution_planner.cpp
    src/research/trace_simulator.cpp
    src/safety/heartbeat.cpp
    src/safety/resource_governor.cpp
    src/safety/runtime_policy.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)

add_library(Vastloom::Core ALIAS vastloom_core)
set_target_properties(vastloom_core PROPERTIES EXPORT_NAME Core)

target_include_directories(vastloom_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(vastloom_core PUBLIC cxx_std_20)
target_link_libraries(vastloom_core PUBLIC Threads::Threads)
vastloom_set_project_warnings(vastloom_core)
vastloom_enable_sanitizers(vastloom_core)

if(WIN32)
    target_compile_definitions(vastloom_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(vastloom_core PRIVATE Psapi)
endif()

add_executable(vastloom
    src/cli/main.cpp
    src/cli/tiny_commands.cpp
)
target_link_libraries(vastloom PRIVATE Vastloom::Core)
target_compile_features(vastloom PRIVATE cxx_std_20)
vastloom_set_project_warnings(vastloom)
vastloom_enable_sanitizers(vastloom)

add_executable(vastloom-watchdog src/watchdog/main.cpp)
target_link_libraries(vastloom-watchdog PRIVATE Vastloom::Core)
target_compile_features(vastloom-watchdog PRIVATE cxx_std_20)
vastloom_set_project_warnings(vastloom-watchdog)
vastloom_enable_sanitizers(vastloom-watchdog)

include(CTest)
if(BUILD_TESTING AND VASTLOOM_BUILD_TESTS)
    add_executable(vastloom_tests
        tests/test_main.cpp
    )
    target_link_libraries(vastloom_tests PRIVATE Vastloom::Core)
    target_compile_features(vastloom_tests PRIVATE cxx_std_20)
    target_compile_definitions(vastloom_tests PRIVATE VASTLOOM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    vastloom_set_project_warnings(vastloom_tests)
    vastloom_enable_sanitizers(vastloom_tests)
    add_test(NAME vastloom_tests COMMAND vastloom_tests)
    add_executable(vastloom_range_artifact_test
        tests/test_range_artifact.cpp
    )
    target_link_libraries(vastloom_range_artifact_test PRIVATE Vastloom::Core)
    target_compile_features(vastloom_range_artifact_test PRIVATE cxx_std_20)
    vastloom_set_project_warnings(vastloom_range_artifact_test)
    vastloom_enable_sanitizers(vastloom_range_artifact_test)
    add_test(NAME vastloom_range_artifact COMMAND vastloom_range_artifact_test)
    if(VASTLOOM_DEEPSEEK_V4_IQ1S_SLICE)
        add_executable(vastloom_iq1s_slice_test tests/test_iq1s_slice.cpp)
        target_link_libraries(vastloom_iq1s_slice_test PRIVATE Vastloom::Core)
        target_compile_features(vastloom_iq1s_slice_test PRIVATE cxx_std_20)
        vastloom_set_project_warnings(vastloom_iq1s_slice_test)
        vastloom_enable_sanitizers(vastloom_iq1s_slice_test)
        add_test(
            NAME vastloom_deepseek_v4_iq1s_slice
            COMMAND vastloom_iq1s_slice_test ${VASTLOOM_DEEPSEEK_V4_IQ1S_SLICE}
        )
        set_tests_properties(vastloom_deepseek_v4_iq1s_slice PROPERTIES
            TIMEOUT 30
            RUN_SERIAL TRUE
        )
    endif()
    if(VASTLOOM_DEEPSEEK_V4_IQ3XXS_SLICE)
        add_executable(vastloom_iq3xxs_slice_test tests/test_iq3xxs_slice.cpp)
        target_link_libraries(vastloom_iq3xxs_slice_test PRIVATE Vastloom::Core)
        target_compile_features(vastloom_iq3xxs_slice_test PRIVATE cxx_std_20)
        vastloom_set_project_warnings(vastloom_iq3xxs_slice_test)
        vastloom_enable_sanitizers(vastloom_iq3xxs_slice_test)
        add_test(
            NAME vastloom_deepseek_v4_iq3xxs_slice
            COMMAND vastloom_iq3xxs_slice_test ${VASTLOOM_DEEPSEEK_V4_IQ3XXS_SLICE}
        )
        set_tests_properties(vastloom_deepseek_v4_iq3xxs_slice PROPERTIES
            TIMEOUT 30
            RUN_SERIAL TRUE
        )
    endif()
    if(VASTLOOM_DEEPSEEK_V4_IQ1S_SLICE AND
       VASTLOOM_DEEPSEEK_V4_IQ1S_UP_SLICE AND
       VASTLOOM_DEEPSEEK_V4_IQ3XXS_SLICE)
        add_executable(vastloom_deepseek_expert_test tests/test_deepseek_expert.cpp)
        target_link_libraries(vastloom_deepseek_expert_test PRIVATE Vastloom::Core)
        target_compile_features(vastloom_deepseek_expert_test PRIVATE cxx_std_20)
        vastloom_set_project_warnings(vastloom_deepseek_expert_test)
        vastloom_enable_sanitizers(vastloom_deepseek_expert_test)
        add_test(
            NAME vastloom_deepseek_v4_real_expert
            COMMAND vastloom_deepseek_expert_test
                ${VASTLOOM_DEEPSEEK_V4_IQ1S_SLICE}
                ${VASTLOOM_DEEPSEEK_V4_IQ1S_UP_SLICE}
                ${VASTLOOM_DEEPSEEK_V4_IQ3XXS_SLICE}
        )
        set_tests_properties(vastloom_deepseek_v4_real_expert PROPERTIES
            TIMEOUT 60
            RUN_SERIAL TRUE
        )
    endif()
    add_test(
        NAME vastloom_cli_tiny_chat
        COMMAND vastloom chat
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/models/vastloom-tinychat-v1.sgm
            --prompt "Quanto fa 2+2?"
            --max-tokens 64
            --no-stream
            --no-stats
    )
    set_tests_properties(vastloom_cli_tiny_chat PROPERTIES PASS_REGULAR_EXPRESSION "2[+]2 fa 4[.]")
    add_test(
        NAME vastloom_cli_tiny_self_test
        COMMAND vastloom self-test
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/models/vastloom-tinychat-v1.sgm
    )
    set_tests_properties(vastloom_cli_tiny_self_test PROPERTIES PASS_REGULAR_EXPRESSION "Result: PASS")

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME vastloom_cli_tiny_integration
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tiny_chat_cli.py
                --vastloom $<TARGET_FILE:vastloom>
                --manifest ${CMAKE_CURRENT_SOURCE_DIR}/examples/models/vastloom-tinychat-v1.sgm
                --vectors ${CMAKE_CURRENT_SOURCE_DIR}/examples/models/vastloom-tinychat-v1.tests.json
        )
        set_tests_properties(vastloom_cli_tiny_integration PROPERTIES TIMEOUT 60)
        add_test(
            NAME vastloom_hf_importer_offline
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_import_hf_model.py
        )
        set_tests_properties(vastloom_hf_importer_offline PROPERTIES TIMEOUT 30)
        add_test(
            NAME vastloom_deepseek_v4_tokenizer
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_deepseek_tokenizer_cli.py
                --vastloom $<TARGET_FILE:vastloom>
        )
        set_tests_properties(vastloom_deepseek_v4_tokenizer PROPERTIES TIMEOUT 60)
        add_test(
            NAME vastloom_deepseek_v4_operator_fixture
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/reference_deepseek_v4_operators.py
                --check
        )
        set_tests_properties(vastloom_deepseek_v4_operator_fixture PROPERTIES TIMEOUT 30)
        add_test(
            NAME vastloom_deepseek_v4_graph_fixture
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/export_deepseek_v4_fixture.py
                --validate-only
        )
        set_tests_properties(vastloom_deepseek_v4_graph_fixture PROPERTIES TIMEOUT 30)
        add_test(
            NAME vastloom_deepseek_v4_gguf_cli
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gguf_cli.py
                --vastloom $<TARGET_FILE:vastloom>
        )
        set_tests_properties(vastloom_deepseek_v4_gguf_cli PROPERTIES TIMEOUT 60)
        add_test(
            NAME vastloom_range_artifact_cli
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_range_artifact_cli.py
                --vastloom $<TARGET_FILE:vastloom>
        )
        set_tests_properties(vastloom_range_artifact_cli PROPERTIES TIMEOUT 30)
        if(VASTLOOM_SMOLLM2_MODEL)
            add_test(
                NAME vastloom_smollm2_reference
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_smollm2_cli.py
                    --vastloom $<TARGET_FILE:vastloom>
                    --model ${VASTLOOM_SMOLLM2_MODEL}
                    --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/smollm2_reference.json
            )
            set_tests_properties(vastloom_smollm2_reference PROPERTIES TIMEOUT 300)
        endif()
        if(VASTLOOM_OLMOE_MODEL)
            add_test(
                NAME vastloom_olmoe_inspection
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_olmoe_cli.py
                    --vastloom $<TARGET_FILE:vastloom>
                    --model ${VASTLOOM_OLMOE_MODEL}
            )
            set_tests_properties(vastloom_olmoe_inspection PROPERTIES TIMEOUT 60)
        endif()
        if(VASTLOOM_OLMOE_FULL_MODEL)
            add_test(
                NAME vastloom_olmoe_runtime_reference
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_olmoe_runtime_cli.py
                    --vastloom $<TARGET_FILE:vastloom>
                    --model ${VASTLOOM_OLMOE_FULL_MODEL}
                    --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/olmoe_reference.json
            )
            set_tests_properties(vastloom_olmoe_runtime_reference PROPERTIES TIMEOUT 1200 RUN_SERIAL TRUE)
        endif()
        if(VASTLOOM_DEEPSEEK_V4_MODEL)
            add_test(
                NAME vastloom_deepseek_v4_inspection
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_deepseek_v4_cli.py
                    --vastloom $<TARGET_FILE:vastloom>
                    --model ${VASTLOOM_DEEPSEEK_V4_MODEL}
            )
            set_tests_properties(vastloom_deepseek_v4_inspection PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
        endif()
        if(VASTLOOM_DEEPSEEK_V4_GGUF_HEADER)
            add_test(
                NAME vastloom_deepseek_v4_gguf_header
                COMMAND $<TARGET_FILE:vastloom> inspect
                    ${VASTLOOM_DEEPSEEK_V4_GGUF_HEADER}
                    --json
            )
            set_tests_properties(vastloom_deepseek_v4_gguf_header PROPERTIES
                TIMEOUT 30
                RUN_SERIAL TRUE
                PASS_REGULAR_EXPRESSION "\\\"architecture\\\":\\\"deepseek4\\\""
            )
        endif()
        if(VASTLOOM_DEEPSEEK_V4_GGUF_CHECKPOINT)
            add_test(
                NAME vastloom_deepseek_v4_gguf_checkpoint
                COMMAND $<TARGET_FILE:vastloom> deepseek-gguf-checkpoint
                    ${VASTLOOM_DEEPSEEK_V4_GGUF_CHECKPOINT}
                    --json
            )
            set_tests_properties(vastloom_deepseek_v4_gguf_checkpoint PROPERTIES
                TIMEOUT 3600
                RUN_SERIAL TRUE
                PASS_REGULAR_EXPRESSION
                    "\"imported_manifest_verified\":true.*\"pinned_0731_checkpoint_spec_verified\":true.*\"reference_config_verified\":true.*\"layer0_reference_catalog_verified\":true.*\"vocabulary_size\":129280.*\"compression_schedule_values\":46.*\"routed_swiglu_schedule_values\":43.*\"shared_swiglu_schedule_values\":43.*\"planned_layers\":43"
            )
            add_test(
                NAME vastloom_deepseek_v4_gguf_probe_plan
                COMMAND $<TARGET_FILE:vastloom> deepseek-gguf-probe-plan
                    ${VASTLOOM_DEEPSEEK_V4_GGUF_CHECKPOINT}
                    --layer 0
                    --experts 0,1,2,3,4,5
                    --json
            )
            set_tests_properties(vastloom_deepseek_v4_gguf_probe_plan PROPERTIES
                TIMEOUT 3600
                RUN_SERIAL TRUE
                PASS_REGULAR_EXPRESSION
                    "\"plan_only\":true.*\"routing_authority_verified\":false.*\"trunk_bytes\":138867160.*\"selected_expert_bytes\":38928384.*\"total_payload_bytes\":177795544.*\"total_range_count\":39.*\"unique_source_objects\":1"
            )
            add_test(
                NAME vastloom_deepseek_v4_range_import_dry_run
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_range_import_cli.py
                    --vastloom $<TARGET_FILE:vastloom>
                    --importer ${CMAKE_CURRENT_SOURCE_DIR}/tools/import_hf_ranges.py
                    --model ${VASTLOOM_DEEPSEEK_V4_GGUF_CHECKPOINT}
            )
            set_tests_properties(vastloom_deepseek_v4_range_import_dry_run PROPERTIES
                TIMEOUT 3600
                RUN_SERIAL TRUE
            )
        endif()
    else()
        message(STATUS "Python3 not found; skipping TinyChat CLI integration test")
    endif()
endif()

install(TARGETS vastloom vastloom-watchdog vastloom_core
    EXPORT VastloomTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/vastloom DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY examples/models/ DESTINATION ${CMAKE_INSTALL_DATADIR}/vastloom/models)
install(FILES LICENSE NOTICE README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/vastloom)
install(FILES third_party/ggml/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/vastloom/third_party/ggml)

configure_package_config_file(
    cmake/VastloomConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/VastloomConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vastloom"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/VastloomConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(EXPORT VastloomTargets
    FILE VastloomTargets.cmake
    NAMESPACE Vastloom::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vastloom"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/VastloomConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/VastloomConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vastloom"
)

set(CPACK_PACKAGE_NAME "vastloom")
set(CPACK_PACKAGE_VENDOR "Vastloom contributors")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION "0.4.0-alpha.4")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_CHECKSUM "SHA256")
set(CPACK_SOURCE_IGNORE_FILES "/build/;/.git/;/.cache/")
set(CPACK_GENERATOR "TGZ;ZIP")
include(CPack)

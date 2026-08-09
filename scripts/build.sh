#!/usr/bin/env sh
set -eu
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVASTLOOM_BUILD_TESTS=ON -DVASTLOOM_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

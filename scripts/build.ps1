$ErrorActionPreference = "Stop"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTARGATE_BUILD_TESTS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure

$ErrorActionPreference = "Stop"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVASTLOOM_BUILD_TESTS=ON -DVASTLOOM_WARNINGS_AS_ERRORS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure

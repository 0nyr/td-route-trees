cmake_minimum_required(VERSION 3.26)
project(td_route_trees LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Local microbenchmarking only: never for shipped artifacts.
option(TRT_NATIVE "Build with -march=native (local benchmarking only)" OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
    cpp/bindings/module.cpp
    cpp/pwlf/pwlf.cpp
    cpp/core/route_eval.cpp
    cpp/structures/baseline.cpp
    cpp/structures/norm.cpp
    cpp/structures/bbt.cpp
    cpp/structures/lca_bst.cpp
    cpp/structures/bench.cpp
)
target_include_directories(_core PRIVATE cpp)
target_compile_definitions(_core PRIVATE TRT_VERSION="${SKBUILD_PROJECT_VERSION_FULL}")
# Bit-identical agreement with the reference checker requires plain IEEE-754
# double arithmetic: no FMA contraction, ever (see cpp/pwlf/FROZEN.md).
target_compile_options(_core PRIVATE -ffp-contract=off)

if(TRT_NATIVE)
    target_compile_options(_core PRIVATE -march=native)
endif()

install(TARGETS _core DESTINATION td_route_trees)

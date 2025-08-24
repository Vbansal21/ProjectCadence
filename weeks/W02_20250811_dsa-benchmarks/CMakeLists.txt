cmake_minimum_required(VERSION 3.20)
project(week02_dsa_benchmarks CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Sensible warnings
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Release flags (documented in README)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(NOT MSVC)
    add_compile_options(-O3 -march=native -DNDEBUG)
  else()
    add_compile_options(/O2 /DNDEBUG)
  endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(bench
  benchmarks/main.cpp
  benchmarks/datasets.hpp
  benchmarks/workloads.hpp
  src/arena.hpp
  src/vector.hpp
  src/list.hpp
  src/heap.hpp
  src/hash.hpp
)

# Linux perf users may also want pthread
if(UNIX)
  target_link_libraries(bench pthread)
endif()

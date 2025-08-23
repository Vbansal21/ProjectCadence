cmake_minimum_required(VERSION 3.20)
project(week01_reverse_stdlib CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SANITIZE "Enable ASan/UBSan for tests" ON)

if(BUILD_SANITIZE)
  add_compile_options(-g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(Catch2)

add_library(mystdlib
  src/my_memcpy.cpp
  src/my_sort.cpp
)

target_include_directories(mystdlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(CTest)
enable_testing()

add_executable(vector_tests tests/vector_tests.cpp)
add_executable(memcpy_tests tests/memcpy_tests.cpp)
add_executable(sort_tests   tests/sort_tests.cpp)

foreach(tgt IN ITEMS vector_tests memcpy_tests sort_tests)
  target_link_libraries(${tgt} PRIVATE mystdlib Catch2::Catch2WithMain)
  add_test(NAME ${tgt} COMMAND ${tgt})
endforeach()

add_executable(bench_insert benchmarks/bench_insert.cpp)
target_link_libraries(bench_insert PRIVATE mystdlib)

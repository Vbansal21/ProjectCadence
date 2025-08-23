#include <chrono>
#include <iostream>
#include <vector>
#include "../src/my_vector.hpp"

static void bench_std(std::size_t N) {
  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<int> v; v.reserve(N);
  for (std::size_t i = 0; i < N; ++i) v.push_back(static_cast<int>(i));
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "std::vector push_back N=" << N << " took "
            << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
}

static void bench_my(std::size_t N) {
  auto t0 = std::chrono::high_resolution_clock::now();
  MyVector<int> v; // implement reserve/ push_back before running
  for (std::size_t i = 0; i < N; ++i) v.push_back(static_cast<int>(i));
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "MyVector push_back N=" << N << " took "
            << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
}

int main() {
  const std::size_t N = 1'000'000;
  bench_std(N);
  bench_my(N); // will abort until implemented
}

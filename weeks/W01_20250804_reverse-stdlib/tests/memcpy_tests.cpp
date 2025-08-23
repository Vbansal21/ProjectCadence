#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <vector>
#include "my_memcpy.hpp"

TEST_CASE("my_memcpy copies bytes exactly for small buffers") {
  alignas(16) unsigned char src[32];
  alignas(16) unsigned char dst[32] = {};
  for (int i = 0; i < 32; ++i) src[i] = static_cast<unsigned char>(i * 7 + 1);
  void* r = my_memcpy(dst, src, sizeof(src));
  REQUIRE(r == dst);
  REQUIRE(std::memcmp(dst, src, sizeof(src)) == 0);
}

TEST_CASE("my_memcpy large buffer (1 MiB)") {
  const std::size_t N = 1u << 20;
  std::vector<unsigned char> src(N), dst(N);
  for (std::size_t i = 0; i < N; ++i) src[i] = static_cast<unsigned char>((i * 131) ^ (i >> 3));
  my_memcpy(dst.data(), src.data(), N);
  REQUIRE(std::memcmp(dst.data(), src.data(), N) == 0);
}

// NOTE: Overlap is UB for memcpy; test memmove separately if you implement it.

#include "my_memcpy.hpp"
#include <cstdio>
#include <cstdlib>

void* my_memcpy(void* dest, const void* src, std::size_t n) noexcept {
  std::fputs("[my_memcpy] TODO: implement (handle alignment & bulk copies)\n", stderr);
  std::abort();
}

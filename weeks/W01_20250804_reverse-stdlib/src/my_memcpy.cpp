#include "my_memcpy.hpp"
#include <cstdint>

void *my_memcpy(void *dest, const void *src, std::size_t n) noexcept
{
  auto *d = static_cast<unsigned char *>(dest);
  const auto *s = static_cast<const unsigned char *>(src);
  for (std::size_t i = 0; i < n; ++i)
    d[i] = s[i];
  return dest;
}

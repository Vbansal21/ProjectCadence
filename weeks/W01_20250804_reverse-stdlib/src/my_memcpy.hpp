#pragma once
#include <cstddef>

// memcpy semantics: no overlap; returns dest.
void* my_memcpy(void* dest, const void* src, std::size_t n) noexcept;

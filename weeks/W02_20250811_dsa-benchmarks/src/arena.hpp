#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <new>
#include <algorithm>

// A simple bump/monotonic arena for experiments.
// Deallocation is a no-op; reset() releases all at once.
// NOTE: This is optional; containers can fall back to new/delete.
class Arena
{
public:
    explicit Arena(std::size_t cap_bytes = 1ull << 26) // 64 MiB default
        : buf_(cap_bytes), head_(0)
    {
    }

    void *allocate(std::size_t bytes, std::size_t align)
    {
        std::size_t p = (head_ + (align - 1)) & ~(align - 1);
        if (p + bytes > buf_.size())
            grow(std::max(bytes, buf_.size()));
        void *out = buf_.data() + p;
        head_ = p + bytes;
        return out;
    }
    void reset() { head_ = 0; }
    std::size_t bytes_used() const { return head_; }
    std::size_t capacity() const { return buf_.size(); }

private:
    void grow(std::size_t extra)
    {
        std::vector<std::uint8_t> nb;
        nb.resize(buf_.size() + extra);
        std::copy(buf_.begin(), buf_.begin() + head_, nb.begin());
        buf_.swap(nb);
    }
    std::vector<std::uint8_t> buf_;
    std::size_t head_;
};

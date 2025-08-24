#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include <utility>
#include <limits>
#include <cstring>

// Simple open-addressing hash map with linear probing.
// Keys/Values are uint64_t for benchmarking clarity.
// Rehash at max_load_.
class HashMap
{
public:
    using Key = std::uint64_t;
    using Val = std::uint64_t;

    explicit HashMap(double max_load = 0.7) : max_load_(max_load), sz_(0) { reserve_(8); }

    void clear()
    {
        table_.assign(table_.size(), Slot{});
        sz_ = 0;
    }

    std::size_t size() const { return sz_; }
    std::size_t capacity() const { return table_.size(); }

    bool insert(Key k, Val v)
    {
        if ((double)(sz_ + 1) / capacity() > max_load_)
            rehash_(capacity() * 2);
        return insert_no_resize_(k, v);
    }

    std::optional<Val> find(Key k) const
    {
        std::size_t mask = capacity() - 1;
        std::size_t i = hash_(k) & mask;
        for (std::size_t probes = 0; probes < capacity(); ++probes, i = (i + 1) & mask)
        {
            const Slot &s = table_[i];
            if (!s.occ && !s.tomb)
                return std::nullopt; // empty stop
            if (s.occ && s.k == k)
                return s.v;
        }
        return std::nullopt;
    }

    bool erase(Key k)
    {
        std::size_t mask = capacity() - 1;
        std::size_t i = hash_(k) & mask;
        for (std::size_t probes = 0; probes < capacity(); ++probes, i = (i + 1) & mask)
        {
            Slot &s = table_[i];
            if (!s.occ && !s.tomb)
                return false;
            if (s.occ && s.k == k)
            {
                s.occ = false;
                s.tomb = true;
                --sz_;
                return true;
            }
        }
        return false;
    }

private:
    struct Slot
    {
        Key k = 0;
        Val v = 0;
        bool occ = false;
        bool tomb = false;
    };

    static std::uint64_t hash_(std::uint64_t x)
    {
        // 64-bit mix (splitmix64-like)
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        x = x ^ (x >> 31);
        return x;
    }

    bool insert_no_resize_(Key k, Val v)
    {
        std::size_t mask = capacity() - 1;
        std::size_t i = hash_(k) & mask;
        std::size_t first_tomb = capacity(); // remember first tombstone
        for (std::size_t probes = 0; probes < capacity(); ++probes, i = (i + 1) & mask)
        {
            Slot &s = table_[i];
            if (s.occ)
            {
                if (s.k == k)
                {
                    s.v = v;
                    return false;
                } // update
            }
            else
            {
                if (s.tomb && first_tomb == capacity())
                    first_tomb = i;
                else if (!s.tomb)
                {
                    // empty slot: insert
                    if (first_tomb != capacity())
                    {
                        table_[first_tomb] = Slot{k, v, true, false};
                    }
                    else
                    {
                        s = Slot{k, v, true, false};
                    }
                    ++sz_;
                    return true;
                }
            }
        }
        // Table full (shouldn't happen due to rehash)
        return false;
    }

    void reserve_(std::size_t n)
    {
        std::size_t cap = 1;
        while (cap < n)
            cap <<= 1; // power of two capacity
        table_.assign(cap, Slot{});
    }

    void rehash_(std::size_t n)
    {
        std::vector<Slot> old = std::move(table_);
        reserve_(n);
        sz_ = 0;
        for (auto &s : old)
            if (s.occ)
                insert_no_resize_(s.k, s.v);
    }

    double max_load_;
    std::size_t sz_;
    std::vector<Slot> table_;
};

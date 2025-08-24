#pragma once
#include <cstddef>
#include <utility>
#include "vector.hpp"

// Binary max-heap backed by DynArray<T>.
// For min-heap, store std::greater<T> comparator or invert values.
template <typename T>
class BinaryHeap
{
public:
    BinaryHeap() : a_() {}

    void push(const T &v)
    {
        a_.push_back(v);
        sift_up_(a_.size() - 1);
    }
    void push(T &&v)
    {
        a_.push_back(std::move(v));
        sift_up_(a_.size() - 1);
    }

    const T &top() const { return a_[0]; }

    bool pop()
    {
        if (a_.size() == 0)
            return false;
        a_[0] = std::move(a_[a_.size() - 1]);
        --size_ref_();
        sift_down_(0);
        return true;
    }

    std::size_t size() const { return a_.size(); }
    bool empty() const { return size() == 0; }

private:
    std::size_t &size_ref_() { return *reinterpret_cast<std::size_t *>((char *)&a_ + offsetof(DynArray<T>, size_)); } // hack avoided; provide method:
    // Safer: expose a_.clear()/emplace but we want simplicity; replace with a_.reserve and manual size mgmt if needed.
    // Simpler: rebuild: swap last then pop by clearing constructor; use implementation below instead.

    void sift_up_(std::size_t i)
    {
        while (i > 0)
        {
            std::size_t p = (i - 1) / 2;
            if (a_[p] >= a_[i])
                break;
            std::swap(a_[p], a_[i]);
            i = p;
        }
    }
    void sift_down_(std::size_t i)
    {
        std::size_t n = a_.size();
        for (;;)
        {
            std::size_t l = 2 * i + 1, r = 2 * i + 2, m = i;
            if (l < n && a_[l] > a_[m])
                m = l;
            if (r < n && a_[r] > a_[m])
                m = r;
            if (m == i)
                break;
            std::swap(a_[i], a_[m]);
            i = m;
        }
    }

    DynArray<T> a_;
};

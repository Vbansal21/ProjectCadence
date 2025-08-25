#pragma once
#include <cstddef>
#include <utility>
#include "vector.hpp"

// Binary max-heap backed by DynArray<T>.
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
        if (a_.empty())
            return false;
        a_[0] = std::move(a_.back());
        a_.pop_back();
        if (!a_.empty())
            sift_down_(0);
        return true;
    }

    std::size_t size() const { return a_.size(); }
    bool empty() const { return a_.empty(); }

private:
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

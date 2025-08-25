#pragma once
#include <cassert>
#include <cstring>
#include <utility>
#include <new>
#include "arena.hpp"

// Minimal dynamic array with growth policy and optional Arena.
template <typename T>
class DynArray
{
public:
    explicit DynArray(float growth = 2.0f, Arena *arena = nullptr)
        : data_(nullptr), size_(0), cap_(0), growth_(growth), arena_(arena) {}

    ~DynArray() { destroy_(); }

    DynArray(const DynArray &o) : DynArray(o.growth_, o.arena_)
    {
        reserve(o.size_);
        for (std::size_t i = 0; i < o.size_; ++i)
            new (data_ + i) T(o.data_[i]);
        size_ = o.size_;
    }

    DynArray(DynArray &&o) noexcept
        : data_(o.data_), size_(o.size_), cap_(o.cap_), growth_(o.growth_), arena_(o.arena_)
    {
        o.data_ = nullptr;
        o.size_ = o.cap_ = 0;
    }

    DynArray &operator=(DynArray o) noexcept
    {
        swap(o);
        return *this;
    }

    void swap(DynArray &o) noexcept
    {
        std::swap(data_, o.data_);
        std::swap(size_, o.size_);
        std::swap(cap_, o.cap_);
        std::swap(growth_, o.growth_);
        std::swap(arena_, o.arena_);
    }

    void reserve(std::size_t n)
    {
        if (n <= cap_)
            return;
        reallocate_(n);
    }

    void push_back(const T &v)
    {
        ensure_cap_(size_ + 1);
        new (data_ + size_) T(v);
        ++size_;
    }
    void push_back(T &&v)
    {
        ensure_cap_(size_ + 1);
        new (data_ + size_) T(std::move(v));
        ++size_;
    }

    template <class... Args>
    T &emplace_back(Args &&...args)
    {
        ensure_cap_(size_ + 1);
        new (data_ + size_) T(std::forward<Args>(args)...);
        return data_[size_++];
    }

    void pop_back()
    {
        assert(size_ > 0);
        data_[size_ - 1].~T();
        --size_;
    }

    T &back()
    {
        assert(size_ > 0);
        return data_[size_ - 1];
    }
    const T &back() const
    {
        assert(size_ > 0);
        return data_[size_ - 1];
    }

    void clear()
    {
        for (std::size_t i = 0; i < size_; ++i)
            data_[i].~T();
        size_ = 0;
    }

    std::size_t size() const { return size_; }
    std::size_t capacity() const { return cap_; }
    bool empty() const { return size_ == 0; }
    T *data() { return data_; }
    const T *data() const { return data_; }

    T &operator[](std::size_t i)
    {
        assert(i < size_);
        return data_[i];
    }
    const T &operator[](std::size_t i) const
    {
        assert(i < size_);
        return data_[i];
    }

private:
    void ensure_cap_(std::size_t need)
    {
        if (need <= cap_)
            return;
        std::size_t ncap = cap_ == 0 ? 1 : static_cast<std::size_t>(cap_ * growth_);
        if (ncap < need)
            ncap = need;
        reallocate_(ncap);
    }

    void reallocate_(std::size_t ncap)
    {
        T *ndata = nullptr;
        if (arena_)
        {
            ndata = static_cast<T *>(arena_->allocate(ncap * sizeof(T), alignof(T)));
            for (std::size_t i = 0; i < size_; ++i)
                new (ndata + i) T(std::move(data_[i]));
            for (std::size_t i = 0; i < size_; ++i)
                data_[i].~T();
        }
        else
        {
            ndata = static_cast<T *>(::operator new[](ncap * sizeof(T), std::align_val_t(alignof(T))));
            for (std::size_t i = 0; i < size_; ++i)
                new (ndata + i) T(std::move(data_[i]));
            for (std::size_t i = 0; i < size_; ++i)
                data_[i].~T();
            ::operator delete[](data_, std::align_val_t(alignof(T)));
        }
        data_ = ndata;
        cap_ = ncap;
    }

    void destroy_()
    {
        if (!data_)
            return;
        for (std::size_t i = 0; i < size_; ++i)
            data_[i].~T();
        if (!arena_)
            ::operator delete[](data_, std::align_val_t(alignof(T)));
        data_ = nullptr;
        size_ = cap_ = 0;
    }

    T *data_;
    std::size_t size_, cap_;
    float growth_;
    Arena *arena_;
};

#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <utility>
#include <iterator>

// Minimal but working MyVector: push_back, reserve, clear, dtor, move ops.
// Copy operations omitted for brevity this week.

template <class T>
class MyVector
{
public:
  using value_type = T;
  using size_type = std::size_t;
  using reference = value_type &;
  using const_reference = const value_type &;
  using pointer = value_type *;
  using const_pointer = const value_type *;
  using iterator = value_type *;
  using const_iterator = const value_type *;

  MyVector() noexcept : data_(nullptr), size_(0), cap_(0) {}
  ~MyVector()
  {
    destroy_range(0, size_);
    ::operator delete(data_);
  }

  MyVector(const MyVector &) = delete;
  MyVector &operator=(const MyVector &) = delete;

  MyVector(MyVector &&other) noexcept
      : data_(other.data_), size_(other.size_), cap_(other.cap_)
  {
    other.data_ = nullptr;
    other.size_ = 0;
    other.cap_ = 0;
  }
  MyVector &operator=(MyVector &&other) noexcept
  {
    if (this != &other)
    {
      destroy_range(0, size_);
      ::operator delete(data_);
      data_ = other.data_;
      size_ = other.size_;
      cap_ = other.cap_;
      other.data_ = nullptr;
      other.size_ = 0;
      other.cap_ = 0;
    }
    return *this;
  }

  size_type size() const noexcept { return size_; }
  size_type capacity() const noexcept { return cap_; }
  bool empty() const noexcept { return size_ == 0; }

  reference operator[](size_type i)
  {
    bounds_check(i);
    return data_[i];
  }
  const_reference operator[](size_type i) const
  {
    bounds_check(i);
    return data_[i];
  }

  reference at(size_type i)
  {
    if (i >= size_)
      throw std::out_of_range("MyVector::at");
    return data_[i];
  }
  const_reference at(size_type i) const
  {
    if (i >= size_)
      throw std::out_of_range("MyVector::at");
    return data_[i];
  }

  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }

  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator cbegin() const noexcept { return data_; }
  iterator end() noexcept { return data_ + size_; }
  const_iterator end() const noexcept { return data_ + size_; }
  const_iterator cend() const noexcept { return data_ + size_; }

  void push_back(const T &v)
  {
    ensure_capacity_for_one();
    ::new (static_cast<void *>(data_ + size_)) T(v);
    ++size_;
  }
  void push_back(T &&v)
  {
    ensure_capacity_for_one();
    ::new (static_cast<void *>(data_ + size_)) T(std::move(v));
    ++size_;
  }

  void reserve(size_type new_cap)
  {
    if (new_cap <= cap_)
      return;
    reallocate(new_cap);
  }

  void clear() noexcept
  {
    destroy_range(0, size_);
    size_ = 0;
  }

private:
  pointer data_;
  size_type size_;
  size_type cap_;

  void bounds_check(size_type i) const
  {
    if (i >= size_)
      throw std::out_of_range("MyVector index");
  }
  void ensure_capacity_for_one()
  {
    if (size_ == cap_)
    {
      size_type new_cap = cap_ ? cap_ * 2 : 1;
      reallocate(new_cap);
    }
  }
  void reallocate(size_type new_cap)
  {
    pointer new_data = static_cast<pointer>(::operator new(sizeof(T) * new_cap));
    for (size_type i = 0; i < size_; ++i)
    {
      ::new (static_cast<void *>(new_data + i)) T(std::move_if_noexcept(data_[i]));
    }
    destroy_range(0, size_);
    ::operator delete(data_);
    data_ = new_data;
    cap_ = new_cap;
  }
  void destroy_range(size_type first, size_type last) noexcept
  {
    for (size_type i = first; i < last; ++i)
    {
      data_[i].~T();
    }
  }
};

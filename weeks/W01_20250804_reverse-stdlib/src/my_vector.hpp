#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <utility>
#include <iterator>

// Minimal, educational vector. Start non-allocating; implement piece by piece.
// Intentionally incomplete: methods abort or throw until you implement them.

template <class T>
class MyVector {
public:
  using value_type      = T;
  using size_type       = std::size_t;
  using reference       = value_type&;
  using const_reference = const value_type&;
  using pointer         = value_type*;
  using const_pointer   = const value_type*;
  using iterator        = value_type*;
  using const_iterator  = const value_type*;

  MyVector() noexcept : data_(nullptr), size_(0), cap_(0) {}
  ~MyVector() { /* implement destroy & deallocate when you add allocation */ }

  MyVector(const MyVector& other) { unimpl_abort("copy ctor"); }
  MyVector(MyVector&& other) noexcept { unimpl_abort("move ctor"); }
  MyVector& operator=(const MyVector& other) { unimpl_abort("copy assign"); }
  MyVector& operator=(MyVector&& other) noexcept { unimpl_abort("move assign"); }

  size_type size() const noexcept { return size_; }
  size_type capacity() const noexcept { return cap_; }
  bool empty() const noexcept { return size_ == 0; }

  reference operator[](size_type i) { bounds_check(i); return data_[i]; }
  const_reference operator[](size_type i) const { bounds_check(i); return data_[i]; }

  reference at(size_type i) { if(i >= size_) throw std::out_of_range("MyVector::at"); return data_[i]; }
  const_reference at(size_type i) const { if(i >= size_) throw std::out_of_range("MyVector::at"); return data_[i]; }

  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }

  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator cbegin() const noexcept { return data_; }
  iterator end() noexcept { return data_ + size_; }
  const_iterator end() const noexcept { return data_ + size_; }
  const_iterator cend() const noexcept { return data_ + size_; }

  void push_back(const T& v) { unimpl_abort("push_back(const T&)"); }
  void push_back(T&& v)      { unimpl_abort("push_back(T&&)"); }

  void reserve(size_type new_cap) { unimpl_abort("reserve"); }
  void resize(size_type n)        { unimpl_abort("resize"); }
  void clear() noexcept           { unimpl_abort("clear"); }

private:
  pointer   data_;
  size_type size_;
  size_type cap_;

  [[noreturn]] static void unimpl_abort(const char* what) {
    std::fprintf(stderr, "[MyVector] implement: %s\n", what);
    std::abort();
  }
  void bounds_check(size_type i) const {
    if (i >= size_) throw std::out_of_range("MyVector index");
  }
};

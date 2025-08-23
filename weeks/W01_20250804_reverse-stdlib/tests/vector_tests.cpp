#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "my_vector.hpp"

TEST_CASE("MyVector<int> push_back grows size and capacity") {
  MyVector<int> v;
  REQUIRE(v.size() == 0);
  const std::size_t N = 128;
  for (int i = 0; i < static_cast<int>(N); ++i) {
    v.push_back(i); // should not reallocate each time
    REQUIRE(v.size() == static_cast<std::size_t>(i + 1));
    REQUIRE(v.capacity() >= v.size());
  }
  // order intact
  for (int i = 0; i < static_cast<int>(N); ++i) {
    REQUIRE(v[i] == i);
  }
}

TEST_CASE("reserve and at() bounds") {
  MyVector<int> v;
  v.reserve(64);
  REQUIRE(v.capacity() >= 64);
  v.push_back(42);
  REQUIRE_THROWS_AS(v.at(1), std::out_of_range);
}

struct Tracked {
  static inline int live = 0;
  int x = 0;
  Tracked() : x(0) { ++live; }
  explicit Tracked(int v) : x(v) { ++live; }
  Tracked(const Tracked& o) : x(o.x) { ++live; }
  Tracked(Tracked&& o) noexcept : x(o.x) { o.x = -1; ++live; }
  Tracked& operator=(const Tracked& o) { x = o.x; return *this; }
  Tracked& operator=(Tracked&& o) noexcept { x = o.x; o.x = -1; return *this; }
  ~Tracked() { --live; }
};

TEST_CASE("copy/move semantics keep counts sane") {
  REQUIRE(Tracked::live == 0);
  {
    MyVector<Tracked> v;
    for (int i = 0; i < 16; ++i) v.push_back(Tracked{i});
    REQUIRE(v.size() == 16);
  }
  REQUIRE(Tracked::live == 0);
}

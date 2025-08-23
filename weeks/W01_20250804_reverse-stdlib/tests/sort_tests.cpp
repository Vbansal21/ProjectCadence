#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>
#include <random>
#include "../src/my_sort.hpp"

TEST_CASE("my_sort sorts ints ascending by default") {
  std::vector<int> v{9,4,6,2,8,5,3,7,1,0};
  my_sort(v.begin(), v.end());
  for (std::size_t i = 1; i < v.size(); ++i) {
    REQUIRE(v[i-1] <= v[i]);
  }
}

struct Item { int key; std::string val; };

TEST_CASE("my_sort with custom comparator") {
  std::vector<Item> v{{3,"c"},{1,"a"},{2,"b"}};
  auto cmp = [](const Item& a, const Item& b){ return a.key > b.key; };
  my_sort(v.begin(), v.end(), cmp);
  REQUIRE(std::is_sorted(v.begin(), v.end(), cmp));
}

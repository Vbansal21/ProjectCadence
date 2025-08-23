#pragma once
#include <functional>
#include <iterator>

template <class RandomIt, class Compare = std::less<>>
void my_sort(RandomIt first, RandomIt last, Compare comp = Compare{});

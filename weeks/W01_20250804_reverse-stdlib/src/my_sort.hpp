#pragma once
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

// Define the template *in the header* so lambdas from tests can instantiate it.

template <class RandomIt, class Compare = std::less<>>
void my_sort(RandomIt first, RandomIt last, Compare comp = Compare{})
{
    using Cat = typename std::iterator_traits<RandomIt>::iterator_category;
    static_assert(std::is_base_of_v<std::random_access_iterator_tag, Cat>,
                  "my_sort requires random-access iterators");
    if (first == last)
        return;

    // Simple stable insertion sort: good enough for tests; optimize later.
    for (RandomIt i = first + 1; i != last; ++i)
    {
        auto key = std::move(*i);
        RandomIt j = i;
        while (j != first && comp(key, *(j - 1)))
        {
            *j = std::move(*(j - 1));
            --j;
        }
        *j = std::move(key);
    }
}

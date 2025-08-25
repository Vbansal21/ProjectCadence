#pragma once
#include <cstdint>
#include <string>
#include <chrono>
#include <iostream>
#include <optional>
#include <numeric>
#include <vector>
#include <algorithm>

#include "datasets.hpp"
#include "../src/vector.hpp"
#include "../src/list.hpp"
#include "../src/heap.hpp"
#include "../src/hash.hpp"

// Minimal checksum sink to prevent dead-code elimination.
struct Sink
{
    volatile std::uint64_t acc = 0;
    void eat(std::uint64_t x) { acc ^= x + 0x9e3779b97f4a7c15ull + (acc << 6) + (acc >> 2); }
};

// timing helper
template <class F>
std::uint64_t time_ns(F &&f)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

struct Row
{
    std::string ds, impl, workload, dist, params;
    std::size_t N;
    int trial;
    std::uint64_t seed, ns;
    std::uint64_t checksum;
};

inline void print_csv_header()
{
    std::cout << "ds,impl,workload,N,dist,params,trial,seed,ns,checksum\n";
}

// ---- Workloads ----

// DynArray: bulk append; then scan.
inline Row run_vector_bulk(std::size_t N, Dist dist, int trial, std::uint64_t seed)
{
    Sink s;
    DynArray<std::uint64_t> a;
    auto keys = (dist == Dist::Uniform) ? gen_uniform(N, seed) : gen_zipf(N, 1.2, seed);

    std::uint64_t ns = time_ns([&]
                               {
        for (auto k: keys) a.push_back(k);
        // scan to prevent DCE
        for (std::size_t i=0;i<a.size();++i) s.eat(a[i]); });

    Row r{"vector", "custom", "bulk_append+scan", dist == Dist::Uniform ? "uniform" : "zipf",
          "growth=2.0", N, trial, seed, ns, s.acc};
    return r;
}

// SinglyList: push_front then scan
inline Row run_list_front(std::size_t N, Dist dist, int trial, std::uint64_t seed)
{
    Sink s;
    SinglyList<std::uint64_t> lst;
    auto keys = (dist == Dist::Uniform) ? gen_uniform(N, seed) : gen_zipf(N, 1.2, seed);

    std::uint64_t ns = time_ns([&]
                               {
        for (auto k: keys) lst.push_front(k);
        lst.for_each([&](auto v){ s.eat(v); }); });

    Row r{"list", "custom", "push_front+scan", dist == Dist::Uniform ? "uniform" : "zipf",
          "", N, trial, seed, ns, s.acc};
    return r;
}

// Heap: push N then pop N
inline Row run_heap_pushpop(std::size_t N, Dist dist, int trial, std::uint64_t seed)
{
    Sink s;
    BinaryHeap<std::uint64_t> h;
    auto keys = (dist == Dist::Uniform) ? gen_uniform(N, seed) : gen_zipf(N, 1.2, seed);

    std::uint64_t ns = time_ns([&]
                               {
        for (auto k: keys) h.push(k);
        while(!h.empty()){ s.eat(h.top()); h.pop(); } });

    Row r{"heap", "custom", "push_then_pop_all", dist == Dist::Uniform ? "uniform" : "zipf",
          "", N, trial, seed, ns, s.acc};
    return r;
}

// HashMap: insert then mixed finds (50% successful)
inline Row run_hash_ops(std::size_t N, Dist dist, int trial, std::uint64_t seed)
{
    Sink s;
    HashMap hm;
    auto keys = (dist == Dist::Uniform) ? gen_uniform(N, seed) : gen_zipf(N, 1.2, seed);
    std::uint64_t ns = time_ns([&]
                               {
        for (auto k: keys) hm.insert(k, k^0xdeadbeefULL);
        // 50/50 successful + unsuccessful queries
        for (std::size_t i=0;i<N;++i){
            auto q = (i%2==0) ? keys[i] : keys[i]^0xabcdef;
            auto r = hm.find(q);
            s.eat(r ? *r : 0x1234ULL);
        } });
    Row r{"hash", "custom", "insert+mixed_find", dist == Dist::Uniform ? "uniform" : "zipf",
          "load<=0.7", N, trial, seed, ns, s.acc};
    return r;
}

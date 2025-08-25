#pragma once
#include <vector>
#include <list>
#include <queue>
#include <unordered_map>
#include <cstdint>
#include <string>
#include "workloads.hpp" // for Row, Sink, Dist, time_ns, generators

// --- std::vector ---
inline Row run_vector_stl_bulk(std::size_t N, Dist dist, int trial, std::uint64_t seed, bool with_reserve)
{
    Sink s;
    std::vector<std::uint64_t> a;
    auto keys = (dist == Dist::Uniform) ? gen_uniform(N, seed) : gen_zipf(N, 1.2, seed);
    if (with_reserve)
        a.reserve(N);
    std::uint64_t ns = time_ns([&]
                               {
        for (auto k: keys) a.push_back(k);
        for (auto v: a) s.eat(v); });
    Row r{"vector", "stl", with_reserve ? "bulk_append+scan(reserve)" : "bulk_append+scan",
          dist == Dist::Uniform ? "uniform" : "zipf", "", N, trial, seed, ns, s.acc};
    return r;
}

// --- std::list ---
inline Row run_list_stl_front(std::size_t N, Dist dist, int trial, std::uint64_t seed)
{
    Sink s;
    std::list<std::uint64_t> lst;
    auto keys = (dist == Dist::Uniform) ? gen_uniform(N, seed) : gen_zipf(N, 1.2, seed);
    std::uint64_t ns = time_ns([&]
                               {
        for (auto k: keys) lst.push_front(k);
        for (auto v: lst) s.eat(v); });
    Row r{"list", "stl", "push_front+scan", dist == Dist::Uniform ? "uniform" : "zipf", "", N, trial, seed, ns, s.acc};
    return r;
}

// --- std::priority_queue ---
inline Row run_heap_stl_pushpop(std::size_t N, Dist dist, int trial, std::uint64_t seed)
{
    Sink s;
    std::priority_queue<std::uint64_t> pq;
    auto keys = (dist == Dist::Uniform) ? gen_uniform(N, seed) : gen_zipf(N, 1.2, seed);
    std::uint64_t ns = time_ns([&]
                               {
        for (auto k: keys) pq.push(k);
        while(!pq.empty()){ s.eat(pq.top()); pq.pop(); } });
    Row r{"heap", "stl", "push_then_pop_all", dist == Dist::Uniform ? "uniform" : "zipf", "", N, trial, seed, ns, s.acc};
    return r;
}

// --- std::unordered_map ---
inline Row run_hash_stl_ops(std::size_t N, Dist dist, int trial, std::uint64_t seed, bool with_reserve)
{
    Sink s;
    std::unordered_map<std::uint64_t, std::uint64_t> m;
    if (with_reserve)
        m.reserve(static_cast<std::size_t>(N / 0.7)); // comparable load factor
    auto keys = (dist == Dist::Uniform) ? gen_uniform(N, seed) : gen_zipf(N, 1.2, seed);
    std::uint64_t ns = time_ns([&]
                               {
        for (auto k: keys) m[k] = (k^0xdeadbeefULL);
        for (std::size_t i=0;i<N;++i){
            auto q = (i%2==0) ? keys[i] : keys[i]^0xabcdefULL;
            auto it = m.find(q);
            s.eat(it==m.end()? 0x1234ULL : it->second);
        } });
    Row r{"hash", "stl", with_reserve ? "insert+mixed_find(reserve)" : "insert+mixed_find",
          dist == Dist::Uniform ? "uniform" : "zipf", "", N, trial, seed, ns, s.acc};
    return r;
}

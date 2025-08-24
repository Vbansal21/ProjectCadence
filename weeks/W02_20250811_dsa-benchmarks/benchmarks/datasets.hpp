#pragma once
#include <vector>
#include <random>
#include <cstdint>
#include <cmath>

enum class Dist
{
    Uniform,
    Zipf
};

inline std::vector<std::uint64_t>
gen_uniform(std::size_t n, std::uint64_t seed = 42)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> d;
    std::vector<std::uint64_t> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        v.push_back(d(rng));
    return v;
}

// Simple Zipf(s) over [1..n] then shuffled into 64-bit keys.
// This is a basic sampler; good enough for benchmarking variation.
inline std::vector<std::uint64_t>
gen_zipf(std::size_t n, double s = 1.2, std::uint64_t seed = 42)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    // Precompute harmonic-like normalizer
    std::vector<double> cdf(n + 1, 0.0);
    for (std::size_t k = 1; k <= n; ++k)
        cdf[k] = cdf[k - 1] + 1.0 / std::pow((double)k, s);
    for (std::size_t k = 1; k <= n; ++k)
        cdf[k] /= cdf[n];

    std::vector<std::uint64_t> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        double u = U(rng);
        std::size_t k = std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin();
        // Mix k with rng to form a 64-bit-ish key
        out.push_back((std::uint64_t)k ^ (rng() << 1));
    }
    return out;
}

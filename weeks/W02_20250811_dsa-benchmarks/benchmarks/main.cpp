#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include "workloads.hpp"

// Simple CLI:
//   ./bench --sizes 1024,4096,16384,65536 --trials 20 --dist uniform
// Dist: uniform | zipf
struct Args
{
    std::vector<std::size_t> sizes{1024, 4096, 16384, 65536, 262144};
    int trials = 10;
    Dist dist = Dist::Uniform;
    std::uint64_t seed0 = 42;
};

Args parse(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        std::string s = argv[i];
        auto next = [&](std::string &out)
        { if (i+1<argc){ out = argv[++i]; } };
        if (s == "--trials")
        {
            std::string v;
            next(v);
            a.trials = std::stoi(v);
        }
        else if (s == "--dist")
        {
            std::string v;
            next(v);
            a.dist = (v == "zipf") ? Dist::Zipf : Dist::Uniform;
        }
        else if (s == "--seed")
        {
            std::string v;
            next(v);
            a.seed0 = std::stoull(v);
        }
        else if (s == "--sizes")
        {
            std::string v;
            next(v);
            a.sizes.clear();
            std::size_t start = 0;
            while (true)
            {
                auto pos = v.find(',', start);
                std::string tok = (pos == std::string::npos) ? v.substr(start) : v.substr(start, pos - start);
                if (!tok.empty())
                    a.sizes.push_back(std::stoull(tok));
                if (pos == std::string::npos)
                    break;
                start = pos + 1;
            }
        }
    }
    return a;
}

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Args a = parse(argc, argv);
    print_csv_header();

    int trial = 0;
    for (int t = 0; t < a.trials; ++t)
    {
        for (auto N : a.sizes)
        {
            std::uint64_t seed = a.seed0 + t * 1315423911ull + N;
            // Vector workload
            auto r1 = run_vector_bulk(N, a.dist, trial, seed);
            std::cout
                << r1.ds << "," << r1.impl << "," << r1.workload << "," << r1.N << "," << r1.dist << "," << r1.params << "," << r1.trial << "," << r1.seed << "," << r1.ns << "," << r1.checksum << "\n";
            // List workload
            auto r2 = run_list_front(N, a.dist, trial, seed + 1);
            std::cout
                << r2.ds << "," << r2.impl << "," << r2.workload << "," << r2.N << "," << r2.dist << "," << r2.params << "," << r2.trial << "," << r2.seed << "," << r2.ns << "," << r2.checksum << "\n";
            // Heap workload (peek style until pop is wired)
            auto r3 = run_heap_pushpop(N, a.dist, trial, seed + 2);
            std::cout
                << r3.ds << "," << r3.impl << "," << r3.workload << "," << r3.N << "," << r3.dist << "," << r3.params << "," << r3.trial << "," << r3.seed << "," << r3.ns << "," << r3.checksum << "\n";
            // Hash workload
            auto r4 = run_hash_ops(N, a.dist, trial, seed + 3);
            std::cout
                << r4.ds << "," << r4.impl << "," << r4.workload << "," << r4.N << "," << r4.dist << "," << r4.params << "," << r4.trial << "," << r4.seed << "," << r4.ns << "," << r4.checksum << "\n";

            ++trial;
        }
    }
    return 0;
}

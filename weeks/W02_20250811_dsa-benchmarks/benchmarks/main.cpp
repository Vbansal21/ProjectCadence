#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "workloads.hpp"
#include "workloads_stl.hpp"

static void print_metadata(){
    std::fprintf(stderr, "# build: %s %s\n", __DATE__, __TIME__);
#if defined(__clang__)
    std::fprintf(stderr, "# compiler: clang %d\n", __clang_major__);
#elif defined(__GNUC__)
    std::fprintf(stderr, "# compiler: gcc %d\n", __GNUC__);
#endif
#ifdef NDEBUG
    std::fprintf(stderr, "# mode: Release\n");
#else
    std::fprintf(stderr, "# mode: Debug\n");
#endif
}

struct Args
{
    std::vector<std::size_t> sizes{128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648};
    int trials = 16;
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
    print_metadata();
    print_csv_header();

    int trial = 0;
    for (int t = 0; t < a.trials; ++t)
    {
        for (auto N : a.sizes)
        {
            std::uint64_t seed = a.seed0 + t * 1315423911ull + N;

            // ---- Custom ----
            auto r1 = run_vector_bulk(N, a.dist, trial, seed);
            std::cout << r1.ds << "," << r1.impl << "," << r1.workload << "," << r1.N << "," << r1.dist << "," << r1.params << "," << r1.trial << "," << r1.seed << "," << r1.ns << "," << r1.checksum << "\n";

            auto r2 = run_list_front(N, a.dist, trial, seed + 1);
            std::cout << r2.ds << "," << r2.impl << "," << r2.workload << "," << r2.N << "," << r2.dist << "," << r2.params << "," << r2.trial << "," << r2.seed << "," << r2.ns << "," << r2.checksum << "\n";

            auto r3 = run_heap_pushpop(N, a.dist, trial, seed + 2);
            std::cout << r3.ds << "," << r3.impl << "," << r3.workload << "," << r3.N << "," << r3.dist << "," << r3.params << "," << r3.trial << "," << r3.seed << "," << r3.ns << "," << r3.checksum << "\n";

            auto r4 = run_hash_ops(N, a.dist, trial, seed + 3);
            std::cout << r4.ds << "," << r4.impl << "," << r4.workload << "," << r4.N << "," << r4.dist << "," << r4.params << "," << r4.trial << "," << r4.seed << "," << r4.ns << "," << r4.checksum << "\n";

            // ---- STL baselines ----
            auto s1 = run_vector_stl_bulk(N, a.dist, trial, seed, false);
            std::cout << s1.ds << "," << s1.impl << "," << s1.workload << "," << s1.N << "," << s1.dist << "," << s1.params << "," << s1.trial << "," << s1.seed << "," << s1.ns << "," << s1.checksum << "\n";

            auto s1r = run_vector_stl_bulk(N, a.dist, trial, seed, true);
            std::cout << s1r.ds << "," << s1r.impl << "," << s1r.workload << "," << s1r.N << "," << s1r.dist << "," << s1r.params << "," << s1r.trial << "," << s1r.seed << "," << s1r.ns << "," << s1r.checksum << "\n";

            auto s2 = run_list_stl_front(N, a.dist, trial, seed + 1);
            std::cout << s2.ds << "," << s2.impl << "," << s2.workload << "," << s2.N << "," << s2.dist << "," << s2.params << "," << s2.trial << "," << s2.seed << "," << s2.ns << "," << s2.checksum << "\n";

            auto s3 = run_heap_stl_pushpop(N, a.dist, trial, seed + 2);
            std::cout << s3.ds << "," << s3.impl << "," << s3.workload << "," << s3.N << "," << s3.dist << "," << s3.params << "," << s3.trial << "," << s3.seed << "," << s3.ns << "," << s3.checksum << "\n";

            auto s4 = run_hash_stl_ops(N, a.dist, trial, seed + 3, false);
            std::cout << s4.ds << "," << s4.impl << "," << s4.workload << "," << s4.N << "," << s4.dist << "," << s4.params << "," << s4.trial << "," << s4.seed << "," << s4.ns << "," << s4.checksum << "\n";

            auto s4r = run_hash_stl_ops(N, a.dist, trial, seed + 4, true);
            std::cout << s4r.ds << "," << s4r.impl << "," << s4r.workload << "," << s4r.N << "," << s4r.dist << "," << s4r.params << "," << s4r.trial << "," << s4r.seed << "," << s4r.ns << "," << s4r.checksum << "\n";

            ++trial;
        }
    }
    return 0;
}

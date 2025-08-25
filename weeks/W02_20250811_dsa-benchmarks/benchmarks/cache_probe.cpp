#include <vector>
#include <chrono>
#include <iostream>
#include <cstdint>
#include <algorithm>

static inline uint64_t now_ns(){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main(){
    std::cout << "bytes,ns_per_elem\n";
    for (size_t bytes = 1<<10; bytes <= (1<<28); bytes <<= 1){
        size_t n = bytes / sizeof(uint64_t);
        std::vector<uint64_t> a(n, 1);
        uint64_t t0=now_ns();
        volatile uint64_t s=0;
        for (int rep=0; rep<8; ++rep){
            for (size_t i=0;i<n;++i) s += a[i];
        }
        uint64_t t1=now_ns();
        double ns_per_elem = double(t1-t0) / (8.0 * n);
        std::cout << bytes << "," << ns_per_elem << "\n";
    }
}

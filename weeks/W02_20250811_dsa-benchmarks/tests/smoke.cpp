#include <cassert>
#include <vector>
#include <list>
#include <queue>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstdint>

#include "../src/vector.hpp"
#include "../src/list.hpp"
#include "../src/heap.hpp"
#include "../src/hash.hpp"

static std::mt19937_64 rng(12345);

// template<class T>
void check_vector() {
    DynArray<std::uint64_t> a;
    std::vector<std::uint64_t> b;
    std::uniform_int_distribution<int> op(0, 2); // 0:push,1:pop,2:mutate
    for (int i=0;i<20000;++i){
        int o = op(rng);
        if (o==0){ // push
            auto x = rng();
            a.push_back(x); b.push_back(x);
        } else if (o==1){ // pop if any
            if (!b.empty()){ a.pop_back(); b.pop_back(); }
        } else { // mutate last if any
            if (!b.empty()){
                auto x = rng();
                a.back() ^= x; b.back() ^= x;
            }
        }
        assert(a.size()==b.size());
        for (size_t k=0;k<a.size();++k) assert(a[k]==b[k]);
    }
}

void check_list() {
    SinglyList<std::uint64_t> a;
    std::list<std::uint64_t> b;
    for (int i=0;i<10000;++i){
        a.push_front(i); b.push_front(i);
    }
    // Compare forward iteration
    auto it=b.begin();
    bool ok=true;
    a.for_each([&](auto v){
        if (it==b.end() || *it!=v) ok=false;
        ++it;
    });
    assert(ok && it==b.end());
    // Pop a few
    for (int i=0;i<100;++i){ bool ra=a.pop_front(); bool rb=(b.size()? (b.pop_front(),true):false); assert(ra==rb); }
}

void check_heap() {
    BinaryHeap<std::uint64_t> h;
    std::priority_queue<std::uint64_t> pq;
    for (int i=0;i<50000;++i){ auto x=rng(); h.push(x); pq.push(x); }
    while(!pq.empty()){
        assert(!h.empty());
        assert(h.top()==pq.top());
        h.pop(); pq.pop();
    }
    assert(h.empty());
}

void check_hash() {
    HashMap hm;
    std::unordered_map<std::uint64_t,std::uint64_t> m;
    std::uniform_int_distribution<int> op(0, 2); // 0:insert,1:erase,2:find
    for (int i=0;i<50000;++i){
        auto k = rng() & ((1ull<<24)-1); // collide a bit
        int o = op(rng);
        if (o==0){
            auto v = k ^ 0x9e37ULL;
            bool ins = hm.insert(k,v);
            m[k]=v;
            (void)ins;
        } else if (o==1){
            bool e1 = hm.erase(k);
            size_t e2 = m.erase(k);
            if (e1) assert(e2==1);
            if (!e1) assert(e2==0);
        } else {
            auto r1 = hm.find(k);
            auto it = m.find(k);
            if (r1.has_value()) { assert(it!=m.end()); assert(*r1==it->second); }
            else { assert(it==m.end()); }
        }
    }
    // spot-check a few
    for (int i=0;i<1000;++i){
        auto k = rng();
        auto r1 = hm.find(k);
        auto it = m.find(k);
        if (r1.has_value()) { assert(it!=m.end()); assert(*r1==it->second); }
        else { assert(it==m.end()); }
    }
}

int main(){
    check_vector();
    check_list();
    check_heap();
    check_hash();
    std::cout << "OK\n";
    return 0;
}

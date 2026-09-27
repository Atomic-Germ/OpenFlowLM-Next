#include "decode_control.hpp"
#include <cassert>
#include <fstream>
#include <limits>

template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}

int main(int argc, char** argv) {
    assert(argc == 2);
    float logits[] = {-5, -2, -2, -7};
    assert(decode_control::greedy(logits,4) == 1);
    rejects([&]{ decode_control::greedy(logits,0); });
    logits[3] = std::numeric_limits<float>::quiet_NaN();
    rejects([&]{ decode_control::greedy(logits,4); });
    logits[3] = std::numeric_limits<float>::infinity();
    rejects([&]{ decode_control::greedy(logits,4); });
    const uint16_t raw[] = {0x0000,0x3f80,0x4000,0xbf80,0x3f00,0xc000};
    { std::ofstream f(argv[1],std::ios::binary); f.write(reinterpret_cast<const char*>(raw),sizeof(raw)); }
    auto row = decode_control::embedding(argv[1],2,3,2);
    assert(row.size()==2 && row[0]==0.5f && row[1]==-2.f);
    row = decode_control::embedding(argv[1],0,3,2);
    assert(row[0]==0.f && row[1]==1.f);
    rejects([&]{ decode_control::embedding(argv[1],3,3,2); });
    rejects([&]{ decode_control::embedding(argv[1],0,3,0); });
    rejects([&]{ decode_control::embedding(argv[1],0,4,2); });
    rejects([&]{ decode_control::embedding(argv[1],0,3,std::numeric_limits<size_t>::max()); });
    return 0;
}

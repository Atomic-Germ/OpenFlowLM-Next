// width + 1 state slots: a pass never overwrites the committed one, so accept(m) only re-points it.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace open_qwen36 {

struct VerifySlots {
    static constexpr unsigned kMaxWidth = 7;
    unsigned width = 0;                 // 0: no verify program
    unsigned cur = 0;                   // the slot holding the committed state
    unsigned perm[kMaxWidth + 1] = {};  // the stream is compiled to read slot 0 and write 1..width; perm maps those to real slots
    size_t n = 0;                       // 0: nothing to accept
    int pos0 = -1;

    void reset() {
        cur = 0;
        n = 0;
        pos0 = -1;
    }

    // a one-position pass is a decode step, which updates the committed slot in place
    void begin(int pos, size_t npos) {
        if (npos < 1 || (npos > 1 && npos > width))
            throw std::runtime_error("verify: " + std::to_string(npos) + " positions, width " + std::to_string(width));
        perm[0] = cur;
        if (npos == 1) {
            perm[1] = cur;
        } else {
            for (unsigned s = 0, j = 1; s <= width; ++s)
                if (s != cur) perm[j++] = s;
        }
        n = npos;
        pos0 = pos;
    }

    int accept(size_t m) {
        if (pos0 < 0 || n == 0) throw std::runtime_error("verify: accept without a verify before it");
        if (m < 1 || m > n)
            throw std::runtime_error("verify: accept(" + std::to_string(m) + ") of a " + std::to_string(n) + "-position pass");
        cur = perm[m];
        const int pos = pos0 + static_cast<int>(m);
        n = 0;
        pos0 = -1;
        return pos;
    }

    // anything but accept() moved the position, so the last pass can no longer be accepted
    void drop() {
        n = 0;
        pos0 = -1;
    }
};

}  // namespace open_qwen36

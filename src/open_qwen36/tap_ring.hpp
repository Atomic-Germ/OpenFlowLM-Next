// Traces: OPEN-DECODE-TAPS (canonical spec: specs/open-engine/spec.md)
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace open_qwen36 {

/// The residuals kept for the drafter: position p's `taps * hidden` floats live in row p % window.
class TapRing {
public:
    void configure(size_t taps, size_t hidden, size_t window) {
        taps_ = taps;
        hidden_ = hidden;
        window_ = taps ? window : 0;
        data_.assign(window_ * width(), 0.f);
        pos_.assign(window_, -1);
        pass_n_ = 0;
        verify_ = false;
    }
    bool on() const { return window_ > 0; }
    size_t width() const { return taps_ * hidden_; }
    size_t window() const { return window_; }
    void clear() { std::fill(pos_.begin(), pos_.end(), -1L); }

    /// A rewind to `pos`: the positions from it on are not the history any more.
    void truncate(long pos) {
        for (long& p : pos_)
            if (p >= pos) p = -1;
    }

    /// Rows [pos0, pos0 + n) are being written; they hold nothing until end().
    void begin(long pos0, size_t n) {
        for (size_t j = 0; j < std::min(n, window_); ++j) pos_[slot(pos0 + static_cast<long>(j))] = -1;
    }
    void put(size_t tap, long pos, const float* row) {
        std::memcpy(data_.data() + slot(pos) * width() + tap * hidden_, row, hidden_ * sizeof(float));
    }
    /// `verify`: the pass is a verify pass, whose rows from m on drop() takes back.
    void end(long pos0, size_t n, bool verify) {
        for (size_t j = 0; j < std::min(n, window_); ++j) pos_[slot(pos0 + static_cast<long>(j))] = pos0 + static_cast<long>(j);
        pass_pos0_ = pos0;
        pass_n_ = n;
        verify_ = verify;
    }
    /// accept(m) after a verify pass: its rows m.. are dead.
    void drop(size_t m) {
        if (!verify_) return;
        for (size_t j = m; j < pass_n_; ++j) {
            long& p = pos_[slot(pass_pos0_ + static_cast<long>(j))];
            if (p == pass_pos0_ + static_cast<long>(j)) p = -1;
        }
        verify_ = false;
    }
    const float* row(long pos) const {
        if (window_ == 0 || pos < 0) return nullptr;
        const size_t s = slot(pos);
        return pos_[s] == pos ? data_.data() + s * width() : nullptr;
    }

private:
    size_t slot(long pos) const { return static_cast<size_t>(pos) % window_; }
    size_t taps_ = 0, hidden_ = 0, window_ = 0;
    std::vector<float> data_;
    std::vector<long> pos_;
    long pass_pos0_ = 0;
    size_t pass_n_ = 0;
    bool verify_ = false;
};

}  // namespace open_qwen36

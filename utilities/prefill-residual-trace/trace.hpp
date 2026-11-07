#pragma once
// Private diagnostic CLI only. Read actual residuals; never write mapped BOs.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace open_qwen36::residual_trace {
class Writer {
    struct Layer { std::ofstream file; size_t rows = 0, hidden = 0; };
    std::string prefix_;
    std::map<int, Layer> layers_;
    bool closed_ = false;
public:
    explicit Writer(std::string prefix = {}) : prefix_(std::move(prefix)) {}
    void append(int layer, size_t start, size_t rows, size_t hidden, const float* x) {
        if (closed_ || prefix_.empty() || layer < 0 || !rows || !hidden || !x ||
            rows > std::numeric_limits<size_t>::max() / hidden / sizeof(float) ||
            start > std::numeric_limits<size_t>::max() - rows)
            throw std::runtime_error("invalid residual trace dimensions or writer state");
        for (size_t i = 0; i < rows * hidden; ++i)
            if (!std::isfinite(x[i])) throw std::runtime_error("non-finite residual trace");
        auto it = layers_.find(layer);
        if (it == layers_.end()) {
            if (start != 0) throw std::runtime_error("residual trace must start at position zero");
            const auto path = prefix_ + "_res" + std::to_string(layer) + ".bin";
            if (std::filesystem::exists(path)) throw std::runtime_error("residual trace already exists: " + path);
            it = layers_.try_emplace(layer).first;
            it->second.hidden = hidden;
            it->second.file.open(path, std::ios::binary | std::ios::out);
        }
        auto& s = it->second;
        if (s.rows != start || s.hidden != hidden)
            throw std::runtime_error("residual trace position gap, duplicate or width change");
        s.file.write(reinterpret_cast<const char*>(x), static_cast<std::streamsize>(rows * hidden * sizeof(float)));
        s.file.flush();
        if (!s.file) throw std::runtime_error("residual trace write failed");
        s.rows += rows;
    }
    void finish(int layers, size_t rows, size_t hidden) {
        if (closed_ || layers <= 0 || !rows || !hidden || layers_.size() != static_cast<size_t>(layers))
            throw std::runtime_error("incomplete residual trace");
        for (int l = 0; l < layers; ++l) {
            auto it = layers_.find(l);
            if (it == layers_.end() || it->second.rows != rows || it->second.hidden != hidden)
                throw std::runtime_error("incomplete residual trace layer");
        }
        for (auto& entry : layers_) {
            entry.second.file.close();
            if (!entry.second.file) throw std::runtime_error("residual trace close failed");
        }
        closed_ = true;
    }
};
inline Writer writer;
} // namespace open_qwen36::residual_trace

// Run the production C++ manifest/config check without loading XRT or weights.
// c++ -std=c++17 -O2 -Isrc -Isrc/include -Iopen_kernels/harness \
//   utilities/check-open-model.cpp src/open_qwen36/manifest.cpp -o /tmp/check-open-model
// /tmp/check-open-model <manifest.json> <config.json>
#include "open_qwen36/manifest.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: check-open-model <manifest.json> <config.json>\n";
        return 2;
    }
    try {
        auto manifest = open_qwen36::Manifest::load(argv[1]);
        nlohmann::json config;
        std::ifstream(argv[2]) >> config;
        manifest.check_model(config, argv[2]);
        std::cout << "PASS: config matches production manifest\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}

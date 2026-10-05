// open_diffusion_reference_tool: reference.hpp's preparation of an edit's input image, as a
// command for specs/open-diffusion/tests/test_reference.py (OPEN-DIFFUSION-REFERENCE).
//
//   open_diffusion_reference_tool IMAGE SIZE OUT.raw
//
// Writes the prepared SIZE x SIZE x 3 RGB8 pixels to OUT.raw and prints describe() on
// stdout; a refused input prints its reason on stderr and exits 3. Host code only (no
// device, no XRT).
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "reference.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: open_diffusion_reference_tool IMAGE SIZE OUT.raw\n");
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    try {
        auto ref = open_diffusion::prepare_reference(bytes.data(), bytes.size(), std::stoi(argv[2]));
        std::ofstream o(argv[3], std::ios::binary);
        o.write(reinterpret_cast<const char*>(ref.rgb.data()), static_cast<std::streamsize>(ref.rgb.size()));
        if (!o) {
            std::fprintf(stderr, "cannot write %s\n", argv[3]);
            return 2;
        }
        std::printf("%s\n", ref.describe().c_str());
    } catch (const open_diffusion::ReferenceError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 3;
    }
    return 0;
}

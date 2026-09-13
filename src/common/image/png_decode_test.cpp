// OPEN-VISION-IMAGE-READ: a PNG decodes to RGB24 whatever FFmpeg is linked.
//
// Usage: png_decode_test <fixture dir> [<oflm-test image dir>]
// The fixtures are written by specs/open-engine/tests/make_png_fixtures.py: one
// .png and one .rgb (the expected tightly packed RGB24) per case. The two real
// images oflm-test bundles are not copied in - bundled.json holds the sha256 of
// the RGB24 Pillow decodes them to, and the second argument says where they live.
#include "image/png_decode.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> v(static_cast<size_t>(n < 0 ? 0 : n));
    if (!v.empty()) f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

void check(bool ok, const std::string& what) {
    std::printf("%-58s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

/// Decode <dir>/<name>.png and compare it byte for byte with <dir>/<name>.rgb.
void expect_matches(const std::string& dir, const std::string& name) {
    const std::vector<uint8_t> png = slurp(dir + "/" + name + ".png");
    const std::vector<uint8_t> want = slurp(dir + "/" + name + ".rgb");
    if (png.empty() || want.empty()) {
        check(false, name + ": fixture missing");
        return;
    }
    int w = 0, h = 0;
    std::vector<uint8_t> got;
    std::string err;
    if (!image_png::decode_rgb24(png.data(), png.size(), w, h, got, err)) {
        check(false, name + ": decode failed (" + err + ")");
        return;
    }
    if (got.size() != want.size()) {
        check(false, name + ": " + std::to_string(got.size()) + " bytes, expected " + std::to_string(want.size()));
        return;
    }
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i]) ++bad;
    check(bad == 0, name + " (" + std::to_string(w) + "x" + std::to_string(h) + ")");
    if (bad) std::printf("    %zu of %zu bytes differ\n", bad, got.size());
}

/// A file this decoder will not take must say so and must not half-fill anything.
void expect_refused(const std::string& dir, const std::string& name, const char* needle) {
    const std::vector<uint8_t> png = slurp(dir + "/" + name + ".png");
    if (png.empty()) {
        check(false, name + ": fixture missing");
        return;
    }
    int w = 0, h = 0;
    std::vector<uint8_t> got;
    std::string err;
    const bool ok = image_png::decode_rgb24(png.data(), png.size(), w, h, got, err);
    check(!ok && err.find(needle) != std::string::npos, name + ": refused with \"" + needle + "\"");
    if (ok) std::printf("    decoded instead of refusing\n");
    else if (err.find(needle) == std::string::npos) std::printf("    said: %s\n", err.c_str());
}

// A tiny SHA-256, so the real images can be checked against Pillow's answer
// without checking 2.4 MB of expected pixels into the tree.
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
                   (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        const uint32_t v[8] = {a, b, c, d, e, f, g, hh};
        for (int i = 0; i < 8; ++i) h[i] += v[i];
    }
    std::string of(const std::vector<uint8_t>& m) {
        std::vector<uint8_t> pad(m);
        const uint64_t bits = static_cast<uint64_t>(m.size()) * 8;
        pad.push_back(0x80);
        while (pad.size() % 64 != 56) pad.push_back(0);
        for (int i = 7; i >= 0; --i) pad.push_back(static_cast<uint8_t>(bits >> (i * 8)));
        for (size_t i = 0; i < pad.size(); i += 64) block(pad.data() + i);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + i * 8, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

/// The real images, checked against the sha256 of Pillow's RGB24 in bundled.json.
void expect_bundled(const std::string& fixdir, const std::string& imgdir, const std::string& name) {
    const std::vector<uint8_t> meta = slurp(fixdir + "/bundled.json");
    if (meta.empty()) { check(false, name + ": bundled.json missing"); return; }
    const std::string j(meta.begin(), meta.end());
    // Small enough to find the fields by hand rather than link a JSON library.
    const size_t at = j.find("\"" + name + "\"");
    if (at == std::string::npos) { check(false, name + ": not in bundled.json"); return; }
    const size_t sk = j.find("\"sha256\"", at);
    const size_t q1 = j.find('"', j.find(':', sk) + 1);
    const std::string want = j.substr(q1 + 1, 64);

    const std::vector<uint8_t> png = slurp(imgdir + "/" + name + ".png");
    if (png.empty()) { check(false, name + ": " + imgdir + "/" + name + ".png missing"); return; }
    int w = 0, h = 0;
    std::vector<uint8_t> got;
    std::string err;
    if (!image_png::decode_rgb24(png.data(), png.size(), w, h, got, err)) {
        check(false, name + ": decode failed (" + err + ")");
        return;
    }
    const std::string have = Sha256().of(got);
    check(have == want, name + " (" + std::to_string(w) + "x" + std::to_string(h) + ") vs Pillow sha256");
    if (have != want) std::printf("    got %s\n    want %s\n", have.c_str(), want.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: png_decode_test <fixture dir> [<oflm-test image dir>]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const std::string imgdir = argc > 2 ? argv[2] : "";

    // Every colour type and bit depth the decoder claims, each fixture written
    // with all five row filters so a wrong predictor cannot pass, and between
    // them all three deflate block types.
    for (const char* n : {"rgb8", "rgba8", "gray8", "graya8", "palette8", "rgb16", "gray16",
                          "gray4", "gray2", "gray1", "palette4", "rgb8_stored", "rgb8_fixed"})
        expect_matches(dir, n);

    // The two images oflm-test actually sends - the ones that were being
    // dropped: 8-bit RGBA, non-interlaced, and spectrogram.png has 69 IDAT
    // chunks plus a private chunk after the last one.
    if (!imgdir.empty()) {
        expect_bundled(dir, imgdir, "paris");
        expect_bundled(dir, imgdir, "spectrogram");
    } else {
        std::printf("%-58s %s\n", "the two bundled images", "skipped (no image dir given)");
    }

    expect_refused(dir, "interlaced", "interlaced");

    // Not a PNG at all, and a PNG cut short mid-IDAT.
    {
        const uint8_t junk[16] = {0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int w = 0, h = 0;
        std::vector<uint8_t> got;
        std::string err;
        check(!image_png::decode_rgb24(junk, sizeof junk, w, h, got, err), "a JPEG is refused");
    }
    expect_refused(dir, "truncated", "PNG");

    std::printf("%s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}

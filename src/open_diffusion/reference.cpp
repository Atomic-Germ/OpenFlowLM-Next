// open_diffusion reference: see reference.hpp.
#include "reference.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../../third_party/stb/stb_image.h"

namespace open_diffusion {

namespace {

constexpr double kPi = 3.14159265358979323846;

// PIL's resample (Pillow's libImaging/Resample.c), which diffusers' Image.resize(...,
// LANCZOS) runs: per output pixel a Lanczos-3 window (scaled by the downscale factor),
// truncated at the image's edges and renormalised, the coefficients in fixed point with
// 22 fraction bits; horizontal then vertical, each pass rounded and clipped to uint8.
constexpr int kPrecisionBits = 32 - 8 - 2;

double lanczos(double x) {
    auto sinc = [](double v) { return v == 0.0 ? 1.0 : std::sin(v * kPi) / (v * kPi); };
    return (-3.0 <= x && x < 3.0) ? sinc(x) * sinc(x / 3.0) : 0.0;
}

struct Coeffs {
    int ksize = 0;
    std::vector<int> bounds;       // per output pixel: first input pixel, count
    std::vector<int32_t> k;        // [out, ksize] fixed point
};

Coeffs precompute(int in_size, int out_size) {
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = 3.0 * filterscale;
    Coeffs c;
    c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    c.bounds.resize(static_cast<size_t>(out_size) * 2);
    c.k.assign(static_cast<size_t>(out_size) * c.ksize, 0);
    std::vector<double> w(c.ksize);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale, ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            w[x] = lanczos((x + xmin - center + 0.5) * ss);
            ww += w[x];
        }
        for (int x = 0; x < xmax; ++x) {
            const double v = ww != 0.0 ? w[x] / ww : w[x];
            c.k[static_cast<size_t>(xx) * c.ksize + x] =
                static_cast<int32_t>(v < 0 ? -0.5 + v * (1 << kPrecisionBits) : 0.5 + v * (1 << kPrecisionBits));
        }
        c.bounds[xx * 2] = xmin;
        c.bounds[xx * 2 + 1] = xmax;
    }
    return c;
}

uint8_t clip8(int32_t v) {
    if (v >= (1 << kPrecisionBits << 8)) return 255;
    if (v <= 0) return 0;
    return static_cast<uint8_t>(v >> kPrecisionBits);
}

// RGB8 [h, w] -> [h, ow] (horizontal) or [oh, w] (vertical)
std::vector<uint8_t> resample(const std::vector<uint8_t>& in, int w, int h, int ow, int oh) {
    const bool horiz = ow != w;
    const Coeffs c = precompute(horiz ? w : h, horiz ? ow : oh);
    std::vector<uint8_t> out(static_cast<size_t>(ow) * oh * 3);
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x) {
            const int o = horiz ? x : y;
            const int first = c.bounds[o * 2], n = c.bounds[o * 2 + 1];
            const int32_t* k = &c.k[static_cast<size_t>(o) * c.ksize];
            int32_t ss[3] = {1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1)};
            for (int i = 0; i < n; ++i) {
                const uint8_t* p = horiz ? &in[(static_cast<size_t>(y) * w + first + i) * 3]
                                         : &in[(static_cast<size_t>(first + i) * w + x) * 3];
                for (int ch = 0; ch < 3; ++ch) ss[ch] += p[ch] * k[i];
            }
            uint8_t* d = &out[(static_cast<size_t>(y) * ow + x) * 3];
            for (int ch = 0; ch < 3; ++ch) d[ch] = clip8(ss[ch]);
        }
    return out;
}

uint16_t be16(const uint8_t* p) { return static_cast<uint16_t>(p[0] << 8 | p[1]); }

// Where pixel (x, y) of the oriented image is in the stored one (EXIF 0x0112, as
// PIL.ImageOps.exif_transpose applies it). w, h: the stored size.
void source_of(int o, int x, int y, int w, int h, int* sx, int* sy) {
    switch (o) {
        case 2: *sx = w - 1 - x; *sy = y; return;
        case 3: *sx = w - 1 - x; *sy = h - 1 - y; return;
        case 4: *sx = x; *sy = h - 1 - y; return;
        case 5: *sx = y; *sy = x; return;
        case 6: *sx = y; *sy = h - 1 - x; return;
        case 7: *sx = w - 1 - y; *sy = h - 1 - x; return;
        case 8: *sx = w - 1 - y; *sy = x; return;
        default: *sx = x; *sy = y; return;
    }
}

}  // namespace

std::string Reference::describe() const {
    std::string s = "reference " + std::to_string(width) + "x" + std::to_string(height);
    if (orientation != 1) s += " (EXIF orientation " + std::to_string(orientation) + " applied)";
    if (width != height) s += " centre-cropped to " + std::to_string(side) + "x" + std::to_string(side);
    if (side != size) {
        s += std::string(side < size ? ", upscaled" : ", resized") + " to " + std::to_string(size) +
             "x" + std::to_string(size);
    }
    return s;
}

int jpeg_orientation(const uint8_t* d, size_t n) {
    if (n < 4 || d[0] != 0xFF || d[1] != 0xD8) return 1;
    size_t i = 2;
    while (i + 4 <= n && d[i] == 0xFF) {
        const uint8_t marker = d[i + 1];
        if (marker == 0xD9 || marker == 0xDA) break;            // end of image, start of scan
        const size_t len = be16(d + i + 2);
        if (len < 2 || i + 2 + len > n) break;
        const uint8_t* seg = d + i + 4;
        const size_t seg_n = len - 2;
        if (marker == 0xE1 && seg_n >= 14 && std::memcmp(seg, "Exif\0\0", 6) == 0) {
            const uint8_t* t = seg + 6;                          // the TIFF header
            const size_t tn = seg_n - 6;
            const bool le = t[0] == 'I' && t[1] == 'I';
            if (!le && !(t[0] == 'M' && t[1] == 'M')) return 1;
            auto u16 = [&](size_t o) -> uint32_t {
                return le ? (t[o] | t[o + 1] << 8) : (t[o] << 8 | t[o + 1]);
            };
            auto u32 = [&](size_t o) -> uint32_t {
                return le ? (t[o] | t[o + 1] << 8 | t[o + 2] << 16 | static_cast<uint32_t>(t[o + 3]) << 24)
                          : (static_cast<uint32_t>(t[o]) << 24 | t[o + 1] << 16 | t[o + 2] << 8 | t[o + 3]);
            };
            const size_t ifd = u32(4);
            if (ifd + 2 > tn) return 1;
            const size_t count = u16(ifd);
            for (size_t e = 0; e < count; e++) {
                const size_t at = ifd + 2 + 12 * e;
                if (at + 12 > tn) return 1;
                if (u16(at) == 0x0112 && u16(at + 2) == 3) {     // SHORT
                    const uint32_t v = u16(at + 8);
                    return v >= 1 && v <= 8 ? static_cast<int>(v) : 1;
                }
            }
            return 1;
        }
        i += 2 + len;
    }
    return 1;
}

bool reference_dims(const uint8_t* data, size_t n, int* width, int* height) {
    int c = 0;
    return data && n > 0 && n <= static_cast<size_t>(INT32_MAX) &&
           stbi_info_from_memory(data, static_cast<int>(n), width, height, &c) != 0;
}

int default_edit_size(int width, int height, const std::vector<int>& sizes) {
    if (sizes.empty()) return 0;
    const int short_side = width < height ? width : height;
    int best = 0, smallest = sizes.front();
    for (int s : sizes) {
        if (s <= short_side && s > best) best = s;
        if (s < smallest) smallest = s;
    }
    return best ? best : smallest;
}

Reference prepare_reference(const uint8_t* data, size_t n, int R) {
    if (!data || n == 0) throw ReferenceError("the reference image is empty");
    if (n > static_cast<size_t>(INT32_MAX)) throw ReferenceError("the reference image is too large");
    const bool png = n >= 8 && std::memcmp(data, "\x89PNG\r\n\x1a\n", 8) == 0;
    const bool jpeg = n >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;
    if (!png && !jpeg) {
        throw ReferenceError("the reference image is not PNG or JPEG (other formats are not implemented)");
    }
    int w = 0, h = 0, c = 0;
    const int len = static_cast<int>(n);
    if (!stbi_info_from_memory(data, len, &w, &h, &c)) {
        throw ReferenceError(std::string("the reference image can't be read: ") + stbi_failure_reason());
    }
    if (static_cast<int64_t>(w) * h > kReferenceMaxPixels) {
        throw ReferenceError("the reference image is " + std::to_string(w) + "x" + std::to_string(h) +
                             ", over the " + std::to_string(kReferenceMaxPixels / 1000000) + " megapixel limit");
    }
    if (w < kReferenceMinSide || h < kReferenceMinSide) {
        throw ReferenceError("the reference image is " + std::to_string(w) + "x" + std::to_string(h) +
                             "; both sides must be at least " + std::to_string(kReferenceMinSide) + " px");
    }
    if (w > static_cast<int64_t>(kReferenceMaxAspect) * h || h > static_cast<int64_t>(kReferenceMaxAspect) * w) {
        throw ReferenceError("the reference image is " + std::to_string(w) + "x" + std::to_string(h) +
                             ", more elongated than " + std::to_string(kReferenceMaxAspect) + ":1");
    }
    Reference ref;
    ref.orientation = jpeg ? jpeg_orientation(data, n) : 1;
    int dw = 0, dh = 0, dc = 0;
    stbi_uc* px = stbi_load_from_memory(data, len, &dw, &dh, &dc, 3);   // RGB: alpha dropped, gray expanded
    if (!px) throw ReferenceError(std::string("the reference image can't be decoded: ") + stbi_failure_reason());
    const bool swap = ref.orientation >= 5;
    ref.width = swap ? dh : dw;
    ref.height = swap ? dw : dh;
    ref.side = ref.width < ref.height ? ref.width : ref.height;
    ref.size = R;
    const int left = (ref.width - ref.side) / 2, top = (ref.height - ref.side) / 2, s = ref.side;
    std::vector<uint8_t> crop(static_cast<size_t>(s) * s * 3);
    for (int y = 0; y < s; y++) {
        for (int x = 0; x < s; x++) {
            int sx = 0, sy = 0;
            source_of(ref.orientation, left + x, top + y, dw, dh, &sx, &sy);
            std::memcpy(&crop[(static_cast<size_t>(y) * s + x) * 3], px + (static_cast<size_t>(sy) * dw + sx) * 3, 3);
        }
    }
    stbi_image_free(px);
    if (s == R) {
        ref.rgb = std::move(crop);
        return ref;
    }
    ref.rgb = resample(resample(crop, s, s, R, s), R, s, R, R);
    return ref;
}

std::vector<uint16_t> reference_bf16(const std::vector<uint8_t>& rgb) {
    std::vector<uint16_t> out(rgb.size());
    for (size_t i = 0; i < rgb.size(); i++) {
        const float v = 2.0f * (static_cast<float>(rgb[i]) / 255.0f) - 1.0f;
        uint32_t b;
        std::memcpy(&b, &v, 4);
        b += 0x7FFF + ((b >> 16) & 1);                          // round to nearest even
        out[i] = static_cast<uint16_t>(b >> 16);
    }
    return out;
}

}  // namespace open_diffusion

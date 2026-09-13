#include "image/png_decode.hpp"

#include <cstring>

namespace image_png {
namespace {

constexpr uint8_t kSignature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};

// ---------------------------------------------------------------- inflate
//
// RFC 1950/1951, enough of it for PNG's IDAT stream and no more. This is here
// rather than calling zlib because linking zlib would give oflm.exe a load-time
// import the Windows installers do not ship: they enumerate DLLs by name and
// list zlib1.dll, while vcpkg's zlib is z.dll. A decoder whose whole point is
// "works whatever is linked" should not add a link dependency to get there.

struct BitReader {
    const uint8_t* p;
    size_t n, pos = 0;
    uint32_t buf = 0;
    int cnt = 0;
    bool bad = false;

    int bit() {
        if (cnt == 0) {
            if (pos >= n) { bad = true; return 0; }
            buf = p[pos++];
            cnt = 8;
        }
        const int b = buf & 1;
        buf >>= 1;
        --cnt;
        return b;
    }
    int bits(int k) {
        int v = 0;
        for (int i = 0; i < k; ++i) v |= bit() << i;
        return v;
    }
    void align() { cnt = 0; buf = 0; }
};

/// Canonical Huffman, decoded a bit at a time (RFC 1951 section 3.2.2).
struct Huff {
    int counts[16] = {0};
    int symbols[288] = {0};
};

bool build(Huff& h, const uint8_t* lengths, int n) {
    for (int i = 0; i < 16; ++i) h.counts[i] = 0;
    for (int i = 0; i < n; ++i) ++h.counts[lengths[i]];
    h.counts[0] = 0;
    int left = 1;
    for (int len = 1; len < 16; ++len) {
        left = (left << 1) - h.counts[len];
        if (left < 0) return false;           // over-subscribed
    }
    int offs[16] = {0};
    for (int len = 1; len < 15; ++len) offs[len + 1] = offs[len] + h.counts[len];
    for (int i = 0; i < n; ++i)
        if (lengths[i]) h.symbols[offs[lengths[i]]++] = i;
    return true;
}

int decode_sym(BitReader& br, const Huff& h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len < 16; ++len) {
        code |= br.bit();
        const int count = h.counts[len];
        if (code - first < count) return h.symbols[index + (code - first)];
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    return -1;
}

const uint16_t kLenBase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                               35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
const uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                               3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const uint16_t kDistBase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
                                257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
                                8193, 12289, 16385, 24577};
const uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                                7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

/// Inflate a raw deflate stream into exactly `out.size()` bytes. Returns false
/// on any malformed input or a length mismatch - PNG knows its unpacked size, so
/// "not exactly this many bytes" is itself corruption.
bool inflate_exact(const uint8_t* src, size_t n, std::vector<uint8_t>& out, std::string& err) {
    BitReader br{src, n};
    size_t o = 0;
    Huff lit, dist;
    for (;;) {
        const int final = br.bit();
        const int type = br.bits(2);
        if (br.bad) { err = "PNG: deflate stream ended early"; return false; }
        if (type == 0) {
            br.align();
            if (br.pos + 4 > br.n) { err = "PNG: stored block header past the end"; return false; }
            const unsigned len = src[br.pos] | (src[br.pos + 1] << 8);
            const unsigned nlen = src[br.pos + 2] | (src[br.pos + 3] << 8);
            br.pos += 4;
            if ((len ^ 0xFFFF) != nlen) { err = "PNG: stored block length check failed"; return false; }
            if (br.pos + len > br.n || o + len > out.size()) { err = "PNG: stored block overruns"; return false; }
            std::memcpy(out.data() + o, src + br.pos, len);
            br.pos += len;
            o += len;
        } else if (type == 1 || type == 2) {
            if (type == 1) {
                uint8_t l[288], d[30];
                for (int i = 0; i < 144; ++i) l[i] = 8;
                for (int i = 144; i < 256; ++i) l[i] = 9;
                for (int i = 256; i < 280; ++i) l[i] = 7;
                for (int i = 280; i < 288; ++i) l[i] = 8;
                for (int i = 0; i < 30; ++i) d[i] = 5;
                if (!build(lit, l, 288) || !build(dist, d, 30)) { err = "PNG: bad fixed tables"; return false; }
            } else {
                static const uint8_t kOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5,
                                                   11, 4, 12, 3, 13, 2, 14, 1, 15};
                const int hlit = br.bits(5) + 257, hdist = br.bits(5) + 1, hclen = br.bits(4) + 4;
                if (hlit > 286 || hdist > 30) { err = "PNG: bad dynamic table sizes"; return false; }
                uint8_t cl[19] = {0};
                for (int i = 0; i < hclen; ++i) cl[kOrder[i]] = static_cast<uint8_t>(br.bits(3));
                Huff clh;
                if (!build(clh, cl, 19)) { err = "PNG: bad code-length table"; return false; }
                uint8_t lengths[318] = {0};
                int i = 0;
                while (i < hlit + hdist) {
                    const int s = decode_sym(br, clh);
                    if (s < 0 || br.bad) { err = "PNG: bad code-length symbol"; return false; }
                    if (s < 16) {
                        lengths[i++] = static_cast<uint8_t>(s);
                    } else {
                        int rep, v = 0;
                        if (s == 16) {
                            if (i == 0) { err = "PNG: repeat with no previous length"; return false; }
                            v = lengths[i - 1];
                            rep = 3 + br.bits(2);
                        } else if (s == 17) {
                            rep = 3 + br.bits(3);
                        } else {
                            rep = 11 + br.bits(7);
                        }
                        if (i + rep > hlit + hdist) { err = "PNG: code lengths overrun"; return false; }
                        while (rep--) lengths[i++] = static_cast<uint8_t>(v);
                    }
                }
                if (!build(lit, lengths, hlit) || !build(dist, lengths + hlit, hdist)) {
                    err = "PNG: bad dynamic tables";
                    return false;
                }
            }
            for (;;) {
                const int s = decode_sym(br, lit);
                if (s < 0 || br.bad) { err = "PNG: bad literal/length symbol"; return false; }
                if (s == 256) break;
                if (s < 256) {
                    if (o >= out.size()) { err = "PNG: more pixel data than the header declares"; return false; }
                    out[o++] = static_cast<uint8_t>(s);
                    continue;
                }
                const int li = s - 257;
                if (li >= 29) { err = "PNG: bad length symbol"; return false; }
                const size_t len = kLenBase[li] + static_cast<size_t>(br.bits(kLenExtra[li]));
                const int di = decode_sym(br, dist);
                if (di < 0 || di >= 30) { err = "PNG: bad distance symbol"; return false; }
                const size_t d = kDistBase[di] + static_cast<size_t>(br.bits(kDistExtra[di]));
                if (d > o) { err = "PNG: back-reference before the start of the stream"; return false; }
                if (o + len > out.size()) { err = "PNG: more pixel data than the header declares"; return false; }
                for (size_t k = 0; k < len; ++k, ++o) out[o] = out[o - d];
            }
        } else {
            err = "PNG: reserved deflate block type";
            return false;
        }
        if (final) break;
    }
    if (br.bad) { err = "PNG: deflate stream ended early"; return false; }
    if (o != out.size()) {
        err = "PNG: got " + std::to_string(o) + " of " + std::to_string(out.size()) + " expected bytes";
        return false;
    }
    return true;
}

/// The two-byte zlib wrapper RFC 1950 puts in front of IDAT.
bool zlib_inflate(const std::vector<uint8_t>& z, std::vector<uint8_t>& out, std::string& err) {
    if (z.size() < 2) { err = "PNG: zlib stream too short"; return false; }
    const unsigned cmf = z[0], flg = z[1];
    if ((cmf & 0x0F) != 8) { err = "PNG: zlib compression method is not deflate"; return false; }
    if (((cmf << 8) | flg) % 31 != 0) { err = "PNG: zlib header check failed"; return false; }
    if (flg & 0x20) { err = "PNG: zlib preset dictionary is not supported"; return false; }
    return inflate_exact(z.data() + 2, z.size() - 2, out, err);
}

uint32_t be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

int channels_for(uint8_t color_type) {
    switch (color_type) {
        case 0: return 1;   // grayscale
        case 2: return 3;   // truecolour
        case 3: return 1;   // palette index
        case 4: return 2;   // grayscale + alpha
        case 6: return 4;   // truecolour + alpha
        default: return 0;
    }
}

int paeth(int a, int b, int c) {
    const int p = a + b - c;
    const int pa = p > a ? p - a : a - p;
    const int pb = p > b ? p - b : b - p;
    const int pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return a;
    return pb <= pc ? b : c;
}

/// Undo the per-scanline filter in place. `raw` is height rows of
/// (1 filter byte + stride bytes); the result is `stride` bytes per row.
bool unfilter(std::vector<uint8_t>& raw, size_t height, size_t stride, size_t bpp, std::string& err) {
    std::vector<uint8_t> out(height * stride);
    const uint8_t* src = raw.data();
    for (size_t y = 0; y < height; ++y) {
        const uint8_t ft = *src++;
        uint8_t* cur = out.data() + y * stride;
        const uint8_t* prev = y ? out.data() + (y - 1) * stride : nullptr;
        std::memcpy(cur, src, stride);
        src += stride;
        switch (ft) {
            case 0:
                break;
            case 1:
                for (size_t i = bpp; i < stride; ++i) cur[i] = static_cast<uint8_t>(cur[i] + cur[i - bpp]);
                break;
            case 2:
                if (prev)
                    for (size_t i = 0; i < stride; ++i) cur[i] = static_cast<uint8_t>(cur[i] + prev[i]);
                break;
            case 3:
                for (size_t i = 0; i < stride; ++i) {
                    const int a = i >= bpp ? cur[i - bpp] : 0;
                    const int b = prev ? prev[i] : 0;
                    cur[i] = static_cast<uint8_t>(cur[i] + ((a + b) >> 1));
                }
                break;
            case 4:
                for (size_t i = 0; i < stride; ++i) {
                    const int a = i >= bpp ? cur[i - bpp] : 0;
                    const int b = prev ? prev[i] : 0;
                    const int c = (prev && i >= bpp) ? prev[i - bpp] : 0;
                    cur[i] = static_cast<uint8_t>(cur[i] + paeth(a, b, c));
                }
                break;
            default:
                err = "PNG: unknown row filter " + std::to_string(ft);
                return false;
        }
    }
    raw.swap(out);
    return true;
}

/// One sample out of a packed scanline, scaled up to 8 bits.
inline uint8_t sample(const uint8_t* row, size_t index, int depth) {
    switch (depth) {
        case 8:  return row[index];
        case 16: return row[index * 2];                  // drop the low byte, as sws does
        case 4:  return static_cast<uint8_t>(((row[index >> 1] >> (index & 1 ? 0 : 4)) & 0x0F) * 17);
        case 2:  return static_cast<uint8_t>(((row[index >> 2] >> (6 - 2 * (index & 3))) & 0x03) * 85);
        default: return static_cast<uint8_t>((row[index >> 3] >> (7 - (index & 7))) & 0x01 ? 255 : 0);
    }
}

}  // namespace

bool decode_rgb24(const uint8_t* data, size_t size, int& width, int& height,
                  std::vector<uint8_t>& rgb, std::string& err) {
    err.clear();
    if (!data || size < 8 || std::memcmp(data, kSignature, 8) != 0) {
        err = "PNG: not a PNG (signature)";
        return false;
    }

    uint32_t w = 0, h = 0;
    uint8_t depth = 0, color_type = 0, interlace = 0;
    bool have_ihdr = false;
    std::vector<uint8_t> palette;   // 3 bytes per entry
    std::vector<uint8_t> idat;

    size_t off = 8;
    while (off + 8 <= size) {
        const uint32_t len = be32(data + off);
        const char* type = reinterpret_cast<const char*>(data + off + 4);
        // len + 12 can overflow only on a length we would reject anyway.
        if (len > size || off + 12 + len > size) {
            err = "PNG: truncated chunk";
            return false;
        }
        const uint8_t* body = data + off + 8;

        if (!std::memcmp(type, "IHDR", 4)) {
            if (len != 13) { err = "PNG: bad IHDR length"; return false; }
            w = be32(body);
            h = be32(body + 4);
            depth = body[8];
            color_type = body[9];
            if (body[10] != 0) { err = "PNG: unknown compression method"; return false; }
            if (body[11] != 0) { err = "PNG: unknown filter method"; return false; }
            interlace = body[12];
            have_ihdr = true;
        } else if (!std::memcmp(type, "PLTE", 4)) {
            palette.assign(body, body + len);
        } else if (!std::memcmp(type, "IDAT", 4)) {
            idat.insert(idat.end(), body, body + len);
        } else if (!std::memcmp(type, "IEND", 4)) {
            break;
        }
        off += 12 + len;
    }

    if (!have_ihdr) { err = "PNG: no IHDR"; return false; }
    if (w == 0 || h == 0) { err = "PNG: zero dimension"; return false; }
    if (interlace != 0) { err = "PNG: interlaced (Adam7) files are not supported"; return false; }
    if (idat.empty()) { err = "PNG: no image data"; return false; }

    const int nch = channels_for(color_type);
    if (nch == 0) { err = "PNG: unknown colour type " + std::to_string(color_type); return false; }
    const bool depth_ok = depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16;
    if (!depth_ok) { err = "PNG: unsupported bit depth " + std::to_string(depth); return false; }
    if (color_type == 3 && palette.empty()) { err = "PNG: palette image with no PLTE"; return false; }
    // The spec's own depth/colour pairings; a file outside them is malformed.
    if ((color_type == 2 || color_type == 4 || color_type == 6) && depth < 8) {
        err = "PNG: colour type " + std::to_string(color_type) + " needs 8- or 16-bit samples";
        return false;
    }
    if (color_type == 3 && depth == 16) { err = "PNG: palette images cannot be 16-bit"; return false; }

    // Guard the allocation before doing it: 1 GB of pixels is already far past
    // anything a vision prompt should carry.
    const uint64_t stride64 = (static_cast<uint64_t>(w) * nch * depth + 7) / 8;
    const uint64_t raw64 = (stride64 + 1) * h;
    if (raw64 > (1ull << 30)) { err = "PNG: image too large"; return false; }

    const size_t stride = static_cast<size_t>(stride64);
    std::vector<uint8_t> raw(static_cast<size_t>(raw64));
    if (!zlib_inflate(idat, raw, err)) return false;

    const size_t bpp = static_cast<size_t>((nch * depth + 7) / 8);   // the filter's "previous pixel" step
    if (!unfilter(raw, h, stride, bpp, err)) return false;

    rgb.assign(static_cast<size_t>(w) * h * 3, 0);
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* row = raw.data() + static_cast<size_t>(y) * stride;
        uint8_t* out = rgb.data() + static_cast<size_t>(y) * w * 3;
        for (uint32_t x = 0; x < w; ++x, out += 3) {
            switch (color_type) {
                case 0:
                    out[0] = out[1] = out[2] = sample(row, x, depth);
                    break;
                case 4:
                    out[0] = out[1] = out[2] = sample(row, static_cast<size_t>(x) * 2, depth);
                    break;
                case 2:
                case 6: {
                    const size_t base = static_cast<size_t>(x) * nch;
                    out[0] = sample(row, base + 0, depth);
                    out[1] = sample(row, base + 1, depth);
                    out[2] = sample(row, base + 2, depth);
                    break;
                }
                case 3: {
                    // sample() scales an index to 0-255; the palette wants the raw one.
                    size_t idx;
                    switch (depth) {
                        case 8: idx = row[x]; break;
                        case 4: idx = (row[x >> 1] >> (x & 1 ? 0 : 4)) & 0x0F; break;
                        case 2: idx = (row[x >> 2] >> (6 - 2 * (x & 3))) & 0x03; break;
                        default: idx = (row[x >> 3] >> (7 - (x & 7))) & 0x01; break;
                    }
                    if (idx * 3 + 2 >= palette.size()) {
                        err = "PNG: palette index " + std::to_string(idx) + " past the end of PLTE";
                        return false;
                    }
                    out[0] = palette[idx * 3 + 0];
                    out[1] = palette[idx * 3 + 1];
                    out[2] = palette[idx * 3 + 2];
                    break;
                }
                default:
                    break;
            }
        }
    }

    width = static_cast<int>(w);
    height = static_cast<int>(h);
    return true;
}

}  // namespace image_png

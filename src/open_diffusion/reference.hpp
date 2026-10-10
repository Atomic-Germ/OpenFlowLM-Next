// open_diffusion reference: an edit's input image, from file bytes to what the NPU encoder
// reads (OPEN-DIFFUSION-REFERENCE; specs/open-diffusion/plans/edits.md, decisions 2 and 3).
//
// Pure host code, once per request, before the first NPU dispatch: decode (PNG and JPEG
// only), apply a JPEG's EXIF orientation, centre-crop to a square, resize it to the output
// size R with a Lanczos-3 filter (diffusers resizes with PIL's LANCZOS), then map to
// bf16 2 (x / 255) - 1 as diffusers' preprocessing does. Inputs are refused, naming why,
// before anything is decoded when the header already says so: not PNG or JPEG, a side
// under 64 px, an aspect over 8:1 (diffusers' limits), or over 64 megapixels.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace open_diffusion {

constexpr int kReferenceMinSide = 64;
constexpr int kReferenceMaxAspect = 8;
constexpr int64_t kReferenceMaxPixels = 64LL * 1000 * 1000;

// What prepare_reference refuses; what() names the reason for the user.
struct ReferenceError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

struct Reference {
    int width = 0, height = 0;   // as decoded and oriented
    int orientation = 1;         // the EXIF orientation applied (1: none)
    int side = 0;                // the centre crop's side, min(width, height)
    int size = 0;                // R: rgb is size x size
    std::vector<uint8_t> rgb;    // [size, size, 3]
    // "reference centre-cropped WxH -> SxS, resized to R" (or "used as is")
    std::string describe() const;
};

// EXIF orientation (1..8) of a JPEG, 1 when it has none or isn't a JPEG.
int jpeg_orientation(const uint8_t* data, size_t n);

// The stored width and height from the file's header, without decoding; false if it is
// not a PNG or JPEG it can read.
bool reference_dims(const uint8_t* data, size_t n, int* width, int* height);

// An edit's default output size (plans/edits.md decision 7): the largest of `sizes` not
// above the reference's shorter side, else the smallest (the reference is then upscaled).
int default_edit_size(int width, int height, const std::vector<int>& sizes);

// The file bytes -> the R x R RGB8 reference. Throws ReferenceError.
Reference prepare_reference(const uint8_t* data, size_t n, int R);

// rgb [R, R, 3] -> bf16 bits of 2 (x / 255) - 1 (float32, round to nearest even), the
// encoder's three input channels.
std::vector<uint16_t> reference_bf16(const std::vector<uint8_t>& rgb);

}  // namespace open_diffusion

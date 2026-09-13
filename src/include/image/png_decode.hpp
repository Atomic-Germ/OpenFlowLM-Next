/// \file png_decode.hpp
/// \brief A PNG decoder that does not go through FFmpeg.
/// \note vcpkg's ffmpeg leaves zlib out of its default features, so the
///       avcodec a fresh Windows clone links has no png decoder at all. This
///       is the reader's own path, so a PNG decodes whatever ffmpeg is around.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace image_png {

/// Decode a PNG into tightly packed RGB24, the format the reader hands on.
///
/// Alpha is dropped rather than composited, which is what the ffmpeg path does
/// (RGBA -> RGB24 through sws_scale). Interlaced files are refused by name.
/// `err` is set on every false return.
bool decode_rgb24(const uint8_t* data, size_t size, int& width, int& height,
                  std::vector<uint8_t>& rgb, std::string& err);

}  // namespace image_png

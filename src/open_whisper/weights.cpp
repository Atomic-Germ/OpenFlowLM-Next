//===- weights.cpp -------------------------------------------*- C++ -*-===//
// open_whisper -- see weights.hpp. SPDX-License-Identifier: MIT
#include "weights.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace ow {

uint16_t bf16_rne(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return static_cast<uint16_t>(((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFFu);
}

// Ported with attribution from NpuEmbeddings' src/open_npue/npue_pack.cpp
// (anonymous-namespace `tile_b`, matching npu_offload/gemm_rtp/npue.py's
// tile_b(order="k,n")). See weights.hpp.
std::vector<uint16_t> tile_b(const float *mat, int64_t K, int64_t N, int64_t tk,
                             int64_t tn, int64_t mac_s, int64_t mac_t) {
  if (K % tk || N % tn)
    throw std::runtime_error("tile_b: [" + std::to_string(K) + "," +
                             std::to_string(N) + "] does not tile into (" +
                             std::to_string(tk) + "," + std::to_string(tn) + ")");
  if (tk % mac_s || tn % mac_t)
    throw std::runtime_error("tile_b: tile (" + std::to_string(tk) + "," +
                             std::to_string(tn) + ") not divisible by mac (" +
                             std::to_string(mac_s) + "," + std::to_string(mac_t) + ")");
  const int64_t kb_n = K / tk, nb_n = N / tn;
  std::vector<uint16_t> out(static_cast<size_t>(K) * static_cast<size_t>(N));
  size_t w = 0;
  for (int64_t kb = 0; kb < kb_n; ++kb)
    for (int64_t nb = 0; nb < nb_n; ++nb)
      for (int64_t si = 0; si < tk / mac_s; ++si)
        for (int64_t ti = 0; ti < tn / mac_t; ++ti)
          for (int64_t s = 0; s < mac_s; ++s)
            for (int64_t t = 0; t < mac_t; ++t) {
              const int64_t r = kb * tk + si * mac_s + s;
              const int64_t c = nb * tn + ti * mac_t + t;
              out[w++] = bf16_rne(mat[r * N + c]);
            }
  return out;
}

namespace {

void require_shape(const open_qwen36::Q4nxFile &f, const std::string &name,
                   const std::vector<size_t> &want) {
  if (!f.has(name))
    throw std::runtime_error("model.open.safetensors: missing tensor '" + name + "'");
  const auto &m = f.meta(name);
  if (m.shape != want) {
    std::string got;
    for (size_t i = 0; i < m.shape.size(); ++i) got += (i ? "," : "") + std::to_string(m.shape[i]);
    std::string exp;
    for (size_t i = 0; i < want.size(); ++i) exp += (i ? "," : "") + std::to_string(want[i]);
    throw std::runtime_error("model.open.safetensors: '" + name + "' has shape [" + got +
                             "], expected [" + exp + "]");
  }
}

std::vector<uint16_t> load_tiled(const open_qwen36::Q4nxFile &f, const std::string &name,
                                 int64_t K, int64_t N, int64_t tk, int64_t tn,
                                 int64_t mac_s, int64_t mac_t) {
  require_shape(f, name, {static_cast<size_t>(K), static_cast<size_t>(N)});
  std::vector<float> fp = f.bf16(name);
  return tile_b(fp.data(), K, N, tk, tn, mac_s, mac_t);
}

std::vector<float> load_f32(const open_qwen36::Q4nxFile &f, const std::string &name,
                            const std::vector<size_t> &shape) {
  require_shape(f, name, shape);
  return f.f32(name);
}

std::string read_file(const std::string &path) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << fs.rdbuf();
  return ss.str();
}

}  // namespace

Weights::Weights(const std::string &model_dir, int64_t tile_k, int64_t tile_n,
                 int64_t mac_s_, int64_t mac_t_)
    : tile_k(tile_k), tile_n(tile_n), mac_s(mac_s_), mac_t(mac_t_) {
  // 1. weights_manifest.json format, checked before opening the safetensors
  //    itself -- a container in a format this reader does not know is a
  //    refusal, not a best-effort parse.
  const std::string manifest_path = model_dir + "/weights_manifest.json";
  const nlohmann::json manifest = nlohmann::json::parse(read_file(manifest_path));
  const std::string format = manifest.value("format", std::string());
  if (format != "oflm-open-whisper-v1")
    throw std::runtime_error("weights_manifest.json: format is '" + format +
                             "', expected 'oflm-open-whisper-v1'");

  // 2. config.json geometry. Refuse anything but whisper-large-v3-turbo's
  //    shape -- the shipped kernel set has no other GEMM streams.
  const nlohmann::json cfg = nlohmann::json::parse(read_file(model_dir + "/config.json"));
  auto want_int = [&](const char *key, int64_t want) {
    if (!cfg.contains(key))
      throw std::runtime_error(std::string("config.json: missing '") + key + "'");
    const int64_t got = cfg.at(key).get<int64_t>();
    if (got != want)
      throw std::runtime_error(std::string("config.json: '") + key + "' is " +
                               std::to_string(got) + ", this engine only implements " +
                               std::to_string(want));
  };
  if (cfg.value("model_type", std::string()) != "whisper")
    throw std::runtime_error("config.json: model_type is not 'whisper'");
  want_int("d_model", Geometry::d_model);
  want_int("encoder_layers", Geometry::n_enc);
  want_int("decoder_layers", Geometry::n_dec);
  want_int("encoder_attention_heads", Geometry::n_heads);
  want_int("encoder_ffn_dim", Geometry::ffn);
  want_int("num_mel_bins", Geometry::n_mel);
  want_int("max_source_positions", Geometry::max_src_pos);

  // 3. The tensors themselves.
  open_qwen36::Q4nxFile f(model_dir + "/model.open.safetensors");
  const int64_t D = Geometry::d_model, FFN = Geometry::ffn;
  const int64_t NDEC = Geometry::n_dec;

  conv1_B = load_tiled(f, "enc.conv1.B", 3 * Geometry::n_mel, D, tile_k, tile_n, mac_s, mac_t);
  conv1_bias = load_f32(f, "enc.conv1.bias", {static_cast<size_t>(D)});
  conv2_B = load_tiled(f, "enc.conv2.B", 3 * D, D, tile_k, tile_n, mac_s, mac_t);
  conv2_bias = load_f32(f, "enc.conv2.bias", {static_cast<size_t>(D)});
  pos = load_f32(f, "enc.pos", {static_cast<size_t>(Geometry::max_src_pos), static_cast<size_t>(D)});
  ln_w = load_f32(f, "enc.ln.w", {static_cast<size_t>(D)});
  ln_b = load_f32(f, "enc.ln.b", {static_cast<size_t>(D)});

  layers.resize(static_cast<size_t>(Geometry::n_enc));
  for (int64_t i = 0; i < Geometry::n_enc; ++i) {
    auto &L = layers[static_cast<size_t>(i)];
    const std::string p = "enc." + std::to_string(i) + ".";
    L.qkv_B = load_tiled(f, p + "qkv.B", D, 3 * D, tile_k, tile_n, mac_s, mac_t);
    L.qkv_bias = load_f32(f, p + "qkv.bias", {static_cast<size_t>(3 * D)});
    L.o_B = load_tiled(f, p + "o.B", D, D, tile_k, tile_n, mac_s, mac_t);
    L.o_bias = load_f32(f, p + "o.bias", {static_cast<size_t>(D)});
    L.fc1_B = load_tiled(f, p + "fc1.B", D, FFN, tile_k, tile_n, mac_s, mac_t);
    L.fc1_bias = load_f32(f, p + "fc1.bias", {static_cast<size_t>(FFN)});
    L.fc2_B = load_tiled(f, p + "fc2.B", FFN, D, tile_k, tile_n, mac_s, mac_t);
    L.fc2_bias = load_f32(f, p + "fc2.bias", {static_cast<size_t>(D)});
    L.ln1_w = load_f32(f, p + "ln1.w", {static_cast<size_t>(D)});
    L.ln1_b = load_f32(f, p + "ln1.b", {static_cast<size_t>(D)});
    L.ln2_w = load_f32(f, p + "ln2.w", {static_cast<size_t>(D)});
    L.ln2_b = load_f32(f, p + "ln2.b", {static_cast<size_t>(D)});
  }

  xkv_B = load_tiled(f, "dec.xkv.B", D, 2 * NDEC * D, tile_k, tile_n, mac_s, mac_t);
  xkv_bias = load_f32(f, "dec.xkv.bias", {static_cast<size_t>(2 * NDEC * D)});
}

}  // namespace ow

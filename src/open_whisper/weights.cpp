//===- weights.cpp -------------------------------------------*- C++ -*-===//
// open_whisper -- see weights.hpp. SPDX-License-Identifier: MIT
#include "weights.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "gguf/gguf_file.hpp"
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

namespace {

// ggml stores a matmul weight [in, out] with `in` fastest; GgufFile::f32_all()
// returns it as HF-natural row-major [out, in]. The encoder's B operand is
// W^T = [in, out], so transpose; the host decoder keeps [out, in] as-is.
std::vector<float> transpose_2d(const std::vector<float>& flat, int64_t rows, int64_t cols) {
  if (static_cast<int64_t>(flat.size()) != rows * cols)
    throw std::runtime_error("whisper gguf: expected " + std::to_string(rows * cols) +
                             " values, got " + std::to_string(flat.size()));
  std::vector<float> out(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c)
      out[static_cast<size_t>(c * rows + r)] = flat[static_cast<size_t>(r * cols + c)];
  return out;
}

// The conv stem: ggml [kernel=3, C_in, C_out] (kernel fastest) -> the
// container's im2col operand [3*C_in, C_out] with K tap-major
// (K = tap * C_in + channel), exactly conv_b() in q4nx-build/open_whisper.py.
std::vector<float> conv_operand(const std::vector<float>& flat, int64_t C_in, int64_t C_out) {
  std::vector<float> out(static_cast<size_t>(3 * C_in * C_out));
  for (int64_t o = 0; o < C_out; ++o)
    for (int64_t c = 0; c < C_in; ++c)
      for (int64_t k = 0; k < 3; ++k)
        out[static_cast<size_t>((k * C_in + c) * C_out + o)] =
            flat[static_cast<size_t>(k + 3 * c + 3 * C_in * o)];
  return out;
}

// One [IN, OUT] tileable B operand from a gguf matmul weight.
std::vector<float> matmul_B(const open_qwen36::GgufFile& gf, const std::string& name,
                            int64_t IN, int64_t OUT) {
  return transpose_2d(gf.f32_all(name), OUT, IN);
}

// qkv / xkv: `names` matrices of [IN, OUT] B operands concatenated along OUT.
std::vector<float> fused_B(const open_qwen36::GgufFile& gf, const std::vector<std::string>& names,
                           int64_t IN, int64_t OUT) {
  const int64_t N = OUT * static_cast<int64_t>(names.size());
  std::vector<float> out(static_cast<size_t>(IN * N));
  int64_t col = 0;
  for (const auto& n : names) {
    const std::vector<float> t = matmul_B(gf, n, IN, OUT);
    for (int64_t r = 0; r < IN; ++r)
      for (int64_t c = 0; c < OUT; ++c)
        out[static_cast<size_t>(r * N + col + c)] = t[static_cast<size_t>(r * OUT + c)];
    col += OUT;
  }
  return out;
}

// A fused bias: `names` give the non-zero halves; the rest are zeros. Each entry
// is [OUT] and the result is [names.size() * OUT]. `zeros` marks a name that is
// an implicit all-zero vector (qkv's k third, xkv's k half).
std::vector<float> fused_bias(const open_qwen36::GgufFile& gf,
                              const std::vector<std::pair<std::string, bool>>& names,
                              int64_t OUT) {
  std::vector<float> out(static_cast<size_t>(OUT * names.size()), 0.0f);
  int64_t col = 0;
  for (const auto& [n, is_zero] : names) {
    if (!is_zero) {
      const std::vector<float> b = gf.f32_all(n);
      if (static_cast<int64_t>(b.size()) != OUT)
        throw std::runtime_error("whisper gguf: bias '" + n + "' is " +
                                 std::to_string(b.size()) + ", expected " + std::to_string(OUT));
      for (int64_t i = 0; i < OUT; ++i) out[static_cast<size_t>(col + i)] = b[static_cast<size_t>(i)];
    }
    col += OUT;
  }
  return out;
}

int64_t gguf_dim(const open_qwen36::GgufFile& gf, const char* key) {
  if (!gf.kv_has(key))
    throw std::runtime_error(std::string("whisper GGUF: missing metadata '") + key + "'");
  return static_cast<int64_t>(gf.kv_u64(key));
}

}  // namespace

Weights Weights::from_gguf(const open_qwen36::GgufFile& gf, int64_t tile_k, int64_t tile_n,
                           int64_t mac_s, int64_t mac_t) {
  // Geometry from the GGUF's own metadata -- refuse anything but
  // whisper-large-v3-turbo's shape, the only one the kernel set serves.
  auto eq = [&](const char* key, int64_t want) {
    const int64_t got = gguf_dim(gf, key);
    if (got != want)
      throw std::runtime_error(std::string("whisper GGUF: '") + key + "' is " +
                               std::to_string(got) + ", this engine only implements " +
                               std::to_string(want));
  };
  eq("stt.whisper.encoder.d_model", Geometry::d_model);
  eq("stt.whisper.encoder.n_layers", Geometry::n_enc);
  eq("stt.whisper.encoder.n_heads", Geometry::n_heads);
  eq("stt.whisper.encoder.ffn_dim", Geometry::ffn);
  eq("stt.whisper.encoder.num_mel_bins", Geometry::n_mel);
  eq("stt.whisper.decoder.d_model", Geometry::d_model);
  eq("stt.whisper.decoder.n_layers", Geometry::n_dec);

  Weights w(tile_k, tile_n, mac_s, mac_t);
  const int64_t D = Geometry::d_model, FFN = Geometry::ffn;
  const int64_t NDEC = Geometry::n_dec;

  auto tile = [&](const std::vector<float>& B, int64_t K, int64_t N) {
    return tile_b(B.data(), K, N, tile_k, tile_n, mac_s, mac_t);
  };

  w.conv1_B = tile(conv_operand(gf.f32_all("enc.conv.0.weight"), Geometry::n_mel, D),
                   3 * Geometry::n_mel, D);
  w.conv1_bias = gf.f32_all("enc.conv.0.bias");
  w.conv2_B = tile(conv_operand(gf.f32_all("enc.conv.1.weight"), D, D), 3 * D, D);
  w.conv2_bias = gf.f32_all("enc.conv.1.bias");
  w.pos = gf.f32_all("enc.pos_emb.weight");          // ggml [d, pos] -> [pos, d] already
  w.ln_w = gf.f32_all("enc.final_norm.weight");
  w.ln_b = gf.f32_all("enc.final_norm.bias");

  w.layers.resize(static_cast<size_t>(Geometry::n_enc));
  for (int64_t i = 0; i < Geometry::n_enc; ++i) {
    auto& L = w.layers[static_cast<size_t>(i)];
    const std::string p = "enc.blocks." + std::to_string(i) + ".";
    L.qkv_B = tile(fused_B(gf, {p + "attn.q.weight", p + "attn.k.weight", p + "attn.v.weight"},
                           D, D),
                   D, 3 * D);
    L.qkv_bias = fused_bias(gf, {{p + "attn.q.bias", false}, {"", true}, {p + "attn.v.bias", false}}, D);
    L.o_B = tile(matmul_B(gf, p + "attn.out.weight", D, D), D, D);
    L.o_bias = gf.f32_all(p + "attn.out.bias");
    L.fc1_B = tile(matmul_B(gf, p + "ffn.fc1.weight", D, FFN), D, FFN);
    L.fc1_bias = gf.f32_all(p + "ffn.fc1.bias");
    L.fc2_B = tile(matmul_B(gf, p + "ffn.fc2.weight", FFN, D), FFN, D);
    L.fc2_bias = gf.f32_all(p + "ffn.fc2.bias");
    L.ln1_w = gf.f32_all(p + "norm_attn.weight");
    L.ln1_b = gf.f32_all(p + "norm_attn.bias");
    L.ln2_w = gf.f32_all(p + "norm_ffn.weight");
    L.ln2_b = gf.f32_all(p + "norm_ffn.bias");
  }

  std::vector<std::string> xnames;
  std::vector<std::pair<std::string, bool>> xbias;
  for (int64_t l = 0; l < NDEC; ++l) {
    const std::string p = "dec.blocks." + std::to_string(l) + ".";
    xnames.push_back(p + "cross_attn.k.weight");
    xnames.push_back(p + "cross_attn.v.weight");
    xbias.emplace_back("", true);
    xbias.emplace_back(p + "cross_attn.v.bias", false);
  }
  w.xkv_B = tile(fused_B(gf, xnames, D, D), D, 2 * NDEC * D);
  w.xkv_bias = fused_bias(gf, xbias, D);

  return w;
}

}  // namespace ow

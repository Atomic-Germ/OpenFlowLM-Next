# Synced from NpuEmbeddings

This directory is a **copy**, written by `tools/sync_openflowlm.py` in
[NpuEmbeddings](https://github.com/vegardberget/NpuEmbeddings). Edit it there,
not here: the accuracy gates that make these numbers mean anything live in
that repository, and a local edit here silently detaches this code from them.

- **Source commit** `b082eb1e6645f129b3caf95e60146a8bffe9b239`
- **Synced** 2026-09-04
- **Licence** MIT (relicensed on copy by the sole copyright holder; upstream is Apache-2.0)

## Toolchain that built the shipped designs

- mlir-aie `1.4.2.dev16+g7e00b57`
- Peano `21.0.0.2026080301+c9c5ecb7`
- mlir-aie HEAD `7e00b57955e108fe9d8e9419f5828a0c7e650858`

## Build notes that are easy to get wrong

- **`/arch:AVX2` (or `-mavx2 -mfma`) is load-bearing.** Roughly half an encode is
  host-side AVX2 intrinsics with correct scalar fallbacks, so without the flag
  it compiles, runs, returns the right answers and is **2.1-2.6x slower**. Set it
  per-source on these files.
- **The embedding bytes are not stable across host ISA levels.** Any check that
  compares vectors against the upstream binary must compile both at the same
  `/arch:` level or it fails for a reason unrelated to the port.
- `npu_device.cpp` needs XRT. Nothing else here does.

## Files

| file | sha256 (after relicensing) | bytes |
|---|---|---:|
| `npue_encoder.cpp` | `606174ba1c3bc706` | 1,581 |
| `npue.cpp` | `ef1b43b9781a3b43` | 9,822 |
| `tokenizer.cpp` | `57a928dd7adafdd6` | 14,509 |
| `npue_pack.cpp` | `71b69d19b7c32bdb` | 93,310 |
| `npu_device.cpp` | `40d1ede49ed314aa` | 21,123 |
| `tokenizer_gemma.cpp` | `c3c9066568a365dc` | 14,498 |
| `gemma_kernels.cpp` | `5c244b39d7cd16b2` | 7,618 |
| `gemma_encode.cpp` | `474d12ff118f317f` | 15,522 |
| `json_min.cpp` | `48c32a087443978f` | 10,398 |
| `gemma_tokenizer_gen.cpp` | `398b836a4417b198` | 13,345 |
| `tokenizer_xlmr.cpp` | `493e464f685e9389` | 21,597 |
| `xlmr_tokenizer_gen.cpp` | `a83d749de2104f3d` | 17,965 |
| `tokenizer_bbpe.cpp` | `6aeab87938278e9e` | 24,941 |
| `bbpe_tokenizer_gen.cpp` | `7b717805983f3cb2` | 14,385 |
| `npue_encoder.hpp` | `5bf7f4c91da9f735` | 222,615 |
| `npue.hpp` | `4baa92b756164cf1` | 4,832 |
| `npue_pack.hpp` | `729cb1933861b989` | 11,200 |
| `npu_device.hpp` | `28f1b733993b038b` | 15,342 |
| `json_min.hpp` | `d347b46053a1f4ba` | 4,506 |
| `tokenizer.hpp` | `70577f60c4f1c31a` | 3,950 |
| `bert_unicode_tables.hpp` | `fe423605b27a69d1` | 622,130 |
| `tokenizer_gemma.hpp` | `ed4ef7da10942108` | 5,265 |
| `gemma_tokenizer_gen.hpp` | `448cc90e15955b3b` | 2,194 |
| `gemma_kernels.hpp` | `32e4b525bd5b419d` | 6,617 |
| `gemma_encode.hpp` | `d44b9c7cc96f2abf` | 4,335 |
| `tokenizer_xlmr.hpp` | `32ae18ee4e82b798` | 5,792 |
| `xlmr_tokenizer_gen.hpp` | `01f64e137efa9589` | 2,265 |
| `xlmr_unicode_tables.hpp` | `72fa6d31ec9e7cc1` | 11,651 |
| `tokenizer_bbpe.hpp` | `6f0ca8d98f0db46a` | 5,233 |
| `bbpe_tokenizer_gen.hpp` | `834d93faac885b9e` | 1,971 |
| `bbpe_unicode_tables.hpp` | `0072df934f77ed43` | 114,763 |
| `npu_offload/gemm_rtp/gemm_pretiled.py` | `8cbcbc8f9cbe7979` | 63,839 |
| `npu_offload/gemm_rtp/export_gemm_rtp.py` | `77410e1ef053b5f3` | 32,238 |
| `npu_offload/gemm_rtp/npue.py` | `10b61d4d839bb2ae` | 22,377 |
| `npu_offload/gemm_rtp/toolchain_provenance.py` | `ab9c09572307d3ef` | 3,419 |
| `npu_offload/m5-eltwise/kernels/narrow_f32_bf16.cc` | `e70669eb7a91fc5a` | 4,947 |
| `npu_offload/m5-eltwise/kernels/narrow_i32_bf16.cc` | `843fd9964f007403` | 5,705 |
| `npu_offload/m5-eltwise/kernels/gelu_poly.cc` | `62f40e41c923f601` | 28,293 |

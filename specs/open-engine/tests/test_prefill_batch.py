# Traces: OPEN-PREFILL-BATCH, OPEN-BUILD-CACHE (canonical spec: specs/open-engine/spec.md)
"""The block prefill route on the 35B: what the qwen36moe recipe writes into the
manifest for it (per layer type a `gemm_block` naming shape-keyed GEMM contexts,
the weight buffers they read and the act offsets the host stages need), and
what it refuses to write (a role the GEMM cannot read)."""
from __future__ import annotations

import dataclasses

import pytest

from recipes import qwen36moe as Q36
from recipes.cache import source_files
from recipes.load import default_spec
from recipes.manifest import manifest
from recipes.spec import FULL, LINEAR

T = 256


@pytest.fixture(scope="module")
def m():
    return manifest(default_spec())


def _run(kernel, w, x, y):
    return {"op": "run", "kernel": kernel, "args": [w, x, y]}


def test_the_35b_linear_layer_type_carries_the_route(m):
    gb = m["layer_types"][LINEAR]["gemm_block"]
    assert gb["kind"] == "linear" and gb["t"] == T and gb["eps"] == 1e-6
    # qkv | z fused (8192 + 4096 rows of K 2048), then out (2048 rows of K 4096); the MoE block,
    # shared expert included, stays on lx1 one token at a time
    assert gb["program"] == [
        _run("gemm_n12288_k2048", "gqkvz_w", "gemm_x_k2048", "gemm_y_n12288"),
        _run("gemm_n2048_k4096", "gout_w", "gemm_x_k4096", "gemm_y_n2048"),
    ]
    assert (gb["qkv_dim"], gb["vw"], gb["key_heads"], gb["value_heads"], gb["head_dim"], gb["conv_kernel"],
            gb["ff"]) == (8192, 4096, 16, 32, 128, 4, 512)
    # the MoE block per token on its own dispatch, patched like lx1
    assert gb["moe_kernel"] == "mx_linear" and gb["moe_args"] == ["pool", "xres", "consts", "state", "act", "ptab"]
    assert m["kernels"]["mx_linear"] == {"context": "mx", "insts": "mx_linear/insts.bin", "patch": "moeroute2",
                                         "build": "mx_linear"}
    assert m["contexts"]["mx"] == "mx_linear/final.xclbin"
    assert m["builds"]["mx_linear"] == {"design": "layer_x/mx.py", "build_dir": "layer_x/build_mx_linear",
                                        "env": {"MX_KIND": "linear"}}
    L = Q36.layout(default_spec())
    assert (gb["a_xm"], gb["a_rout"], gb["a_res"]) == (L.A_XM, L.A_ROUT, L.A_RES)
    assert (gb["state_s_off"], gb["s_head_bytes"], gb["s_rows"]) == (L.STATE_S_OFF, L.S_HEAD_BYTES, L.S_ROWS)
    # each weight buffer is a run of pack ops, named so the driver can check the regions are contiguous
    pool = m["layer_types"][LINEAR]["pack"]["pool"]
    consts = m["layer_types"][LINEAR]["pack"]["consts"]
    w = gb["weights"]
    assert w["gqkvz_w"] == {"from": "pool", "ops": [5, 6]}
    assert [pool[i]["tensor"].split(".")[-2] for i in w["gqkvz_w"]["ops"]] == ["qkv_proj", "gate_proj"]
    assert w["gout_w"] == {"from": "consts", "ops": [len(consts) - 1]}
    assert consts[w["gout_w"]["ops"][0]]["tensor"].endswith("ssm_out_proj.weight")
    assert [pool[i]["op"] for i in (5, 6)] == ["std_perm"] * 2 and consts[w["gout_w"]["ops"][0]]["op"] == "std_perm"


def test_the_35b_attention_layer_type_carries_the_route(m):
    gb = m["layer_types"][FULL]["gemm_block"]
    assert gb["kind"] == "full" and gb["t"] == T
    assert gb["program"] == [
        _run("gemm_n9216_k2048", "gqkvg_w", "gemm_x_k2048", "gemm_y_n9216"),
        _run("gemm_n2048_k4096", "go_w", "gemm_x_k4096", "gemm_y_n2048"),
    ]
    assert (gb["qw"], gb["kvw"], gb["nh"], gb["kvh"], gb["hd"], gb["rot"], gb["ff"]) == (4096, 512, 16, 2, 256, 64, 512)
    assert gb["moe_kernel"] == "mx_full" and m["kernels"]["mx_full"]["patch"] == "moeroute2"
    assert m["kernels"]["mx_full"]["context"] == "mx", "both MoE streams share one xclbin"
    assert m["builds"]["mx_full"]["env"] == {"MX_KIND": "full"}
    L = Q36.layout(default_spec())
    assert (gb["a_xm"], gb["a_rout"], gb["a_res"]) == (L.AA_XM, L.AA_ROUT, L.AA_RES)
    w = gb["weights"]
    assert w["gqkvg_w"] == {"from": "pool", "ops": [5, 6, 7, 8]} and w["go_w"] == {"from": "pool", "ops": [9]}
    pool = m["layer_types"][FULL]["pack"]["pool"]
    assert [pool[i]["tensor"].split(".")[-2] for i in (5, 6, 7, 8, 9)] == ["q_proj", "k_proj", "v_proj", "q_proj", "o_proj"]


def test_the_gemm_route_is_one_context_and_a_stream_per_shape(m):
    # the core program depends on neither N nor K, so every shape is a stream over one xclbin
    assert [k for k in m["contexts"] if k == "gemm" or k.startswith("gemm_")] == ["gemm"]
    # the context points at the first build, the shapes being emitted in sorted order
    assert m["contexts"]["gemm"] == "gemm_n1024_k2048/final.xclbin"
    names = sorted(k for k in m["kernels"] if k.startswith("gemm_"))
    assert names == ["gemm_n1024_k2048", "gemm_n12288_k2048", "gemm_n2048_k4096",
                     "gemm_n2048_k512", "gemm_n9216_k2048"]
    for n in names:
        N, K = (int(p[1:]) for p in n.split("_")[1:])
        assert N % 256 == 0 and K % 256 == 0
        assert m["kernels"][n] == {"context": "gemm", "insts": f"{n}/insts.bin", "build": n}
        b = m["builds"][n]
        assert b["design"] == "gemm_q4_prefill/gemm_q4_prefill.py"
        assert b["build_dir"] == f"gemm_q4_prefill/build_n{N}_k{K}_t{T}"
        assert b["env"] == {"GQP_N": str(N), "GQP_K": str(K), "GQP_T": str(T)}
    g = m["globals"]
    assert g["gemm_x_k2048"] == 2048 * T * 2 and g["gemm_x_k4096"] == 4096 * T * 2
    assert g["gemm_y_n12288"] == 12288 * T * 4 and g["gemm_y_n9216"] == 9216 * T * 4 and g["gemm_y_n2048"] == 2048 * T * 4
    # every step of every gemm_block names a declared kernel, and every kernel a context
    for lt in (LINEAR, FULL):
        for s in m["layer_types"][lt]["gemm_block"]["program"]:
            assert s["kernel"] in m["kernels"] and m["kernels"][s["kernel"]]["context"] in m["contexts"]


def test_a_q8_role_emits_no_route():
    """The GEMM dequantises the q4_1 pool law; a projection the kernel set streams at q8
    (the 35B fine-tunes' attention / linear roles) has no route, and its manifest is
    exactly the sequential one."""
    q8 = dataclasses.replace(default_spec(), quant={"attn": "q8", "linear": "q8", "linear_out": "q8"})
    m = manifest(q8)
    for lt in (LINEAR, FULL):
        assert "gemm_block" not in m["layer_types"][lt]
    assert not [k for k in m["contexts"] if k.startswith("gemm") or k.startswith("mx_")]
    assert not [k for k in m["globals"] if k.startswith("gemm_")]
    assert not [k for k in m["builds"] if k.startswith("gemm_") or k.startswith("mx_")]


def test_the_sequential_program_is_untouched(m):
    for lt, k in ((LINEAR, "lx"), (FULL, "ax")):
        assert [s["op"] for s in m["layer_types"][lt]["program"]] == ["run", "moeroute2", "run"]
        assert m["layer_types"][lt]["program"][0]["kernel"] == k + "0"


def test_the_build_key_covers_the_gemm_sources():
    files = [f.as_posix() for f in source_files(default_spec())]
    for must in ("designs/gemm_q4_prefill/gemm_q4_prefill.py", "designs/gemm_q4_prefill/gemm_q4_dequant.h",
                 "designs/gemm_q4_prefill/gemm_q4_dequant_entry_k0.cc"):
        assert any(f.endswith(must) for f in files), must


# ---- the host stages' numpy reference (open_kernels/model/replica_block.py): a block equals
# the same tokens one at a time with the state carried, padding past t_real changes nothing

import numpy as np  # noqa: E402

import replica_block as RB  # noqa: E402


@pytest.fixture(scope="module")
def case():
    return RB.random_case(1)


def _deltanet(c, qkv, z, xn, conv_state, S, t_real=None):
    return RB.deltanet_block(qkv, z, xn, c["convw"], c["Wa"], c["Wb"], c["A"], c["dtb"], c["nw"], conv_state, S,
                             key_heads=c["key_heads"], value_heads=c["value_heads"], head_dim=c["head_dim"],
                             t_real=t_real)


def test_deltanet_block_equals_one_token_at_a_time(case):
    c = case
    cs_a, S_a = c["conv_state"].copy(), c["S"].copy()
    og_block = _deltanet(c, c["qkv"], c["z"], c["xn"], cs_a, S_a)
    cs_b, S_b = c["conv_state"].copy(), c["S"].copy()
    og_seq = np.vstack([_deltanet(c, c["qkv"][t:t + 1], c["z"][t:t + 1], c["xn"][t:t + 1], cs_b, S_b)
                        for t in range(c["T"])])
    np.testing.assert_allclose(og_block, og_seq, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(cs_a, cs_b)
    np.testing.assert_allclose(S_a, S_b, rtol=1e-12, atol=1e-12)
    assert np.all(S_a[:, c["head_dim"]:] == 0), "the padded S rows stay zero"


def test_deltanet_padding_past_t_real_changes_nothing(case):
    c = case
    cs_a, S_a = c["conv_state"].copy(), c["S"].copy()
    og_full = _deltanet(c, c["qkv"], c["z"], c["xn"], cs_a, S_a, t_real=c["t_real"])
    cs_b, S_b = c["conv_state"].copy(), c["S"].copy()
    og_cut = _deltanet(c, c["qkv"][:c["t_real"]], c["z"][:c["t_real"]], c["xn"][:c["t_real"]], cs_b, S_b)
    np.testing.assert_array_equal(og_full[:c["t_real"]], og_cut)
    assert np.all(og_full[c["t_real"]:] == 0)
    np.testing.assert_array_equal(cs_a, cs_b)
    np.testing.assert_array_equal(S_a, S_b)


def test_attention_block_equals_one_token_at_a_time(case):
    c = case
    kv_a = c["kv"].copy()
    og_block = RB.attention_block(c["q"], c["k"], c["v"], c["gate"], c["qn"], c["kn"], c["inv_freq"], kv_a, c["pos0"],
                                  nh=c["nh"], kvh=c["kvh"], hd=c["hd"])
    kv_b = c["kv"].copy()
    og_seq = np.vstack([RB.attention_block(c["q"][t:t + 1], c["k"][t:t + 1], c["v"][t:t + 1], c["gate"][t:t + 1],
                                           c["qn"], c["kn"], c["inv_freq"], kv_b, c["pos0"] + t,
                                           nh=c["nh"], kvh=c["kvh"], hd=c["hd"]) for t in range(c["T"])])
    np.testing.assert_allclose(og_block, og_seq, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(kv_a, kv_b)
    assert np.all(kv_a[:c["pos0"]] == c["kv"][:c["pos0"]]), "rows before the block are untouched"
    assert np.all(kv_a[c["pos0"]:] == RB.bf16_round(kv_a[c["pos0"]:])), "the cache rows are bf16"


def test_router_block_picks_the_top_k_lowest_index_on_a_tie():
    xm = np.array([[1.0, 0.0]])
    Wr = np.array([[0.5, 0.5, 0.2, 0.5], [0.0, 0.0, 0.0, 0.0]])          # experts 0, 1, 3 tie at the top
    p, idx, w = RB.router_block(xm, Wr, 2)
    np.testing.assert_allclose(p.sum(-1), 1.0)
    assert idx.tolist() == [[0, 1]] and idx.dtype == np.int32
    np.testing.assert_allclose(w, [[0.5, 0.5]])


def test_fixture_round_trips(tmp_path):
    case = RB.random_case(2)
    RB.write_fixture(tmp_path, case, RB.run_case(case))
    shapes = dict(line.split() for line in (tmp_path / "shapes.txt").read_text().splitlines())
    assert shapes["T"] == "6" and (tmp_path / "out_og_lin.f32").stat().st_size == 6 * 4 * 8 * 4
    assert (tmp_path / "inv_freq.f64").stat().st_size == 4 * 8

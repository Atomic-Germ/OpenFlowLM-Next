"""The name-map builder: {bid} detection, passthrough, reverse coverage.

`_create_name_maps` is the converter's routing table and had no test at all,
which is how a whole class of silent container bugs stays alive: a template that
matches nothing, a layer range inferred from a *different* pattern, a config
entry the GGUF never supplied. All of it is pure string work over a fake tensor
index, so none of it needs a model file.

The per-pattern layer count is the subtle one. `num_layers = max(found) + 1` is
computed independently for each distinct `{bid}` template, so two patterns can
legitimately disagree -- and when they do, the shorter one silently drops its
tail layers instead of failing.
"""
import re

import pytest

from q4nx.model_converter import (
    _WarnDict,
    _warn_missing_config_entries,
    config_coverage_report,
)


# --- _WarnDict: a speculative pack must not crash on an unknown name ---

class TestWarnDict:
    def test_unknown_key_passes_through_unchanged(self):
        d = _WarnDict({"known": 1})
        assert d["not_in_the_map"] == "not_in_the_map"

    def test_unknown_is_reported_once_per_name(self):
        d = _WarnDict({"known": 1})
        d["ghost"]
        d["ghost"]
        d["ghost"]
        assert d.unknown == ["ghost"]

    def test_known_keys_are_not_reported(self):
        d = _WarnDict({"known": 1})
        d["known"]
        assert d.unknown == []

    def test_missing_value_overrides_passthrough(self):
        # tensor_q4nx_type_map passes missing_value=self.default_tensor_type so
        # an untyped tensor falls back to the family default rather than keeping
        # its own name as a "type".
        d = _WarnDict({"known": "Q8_0"}, missing_value="Q4_1", missing_label="tensor type")
        assert d["untyped"] == "Q4_1"
        assert d.unknown == ["untyped"]

    def test_is_still_a_dict(self):
        d = _WarnDict({"a": 1})
        d["b"]
        assert set(d.keys()) == {"a"}
        assert len(d) == 1


# --- _warn_missing_config_entries: reverse coverage ---

class TestMissingConfigEntries:
    def test_reports_a_template_the_gguf_never_supplied(self):
        name_map = {
            "q": {"gguf_name": "blk.{bid}.attn_q.weight",
                  "q4nx_name": "model.layers.{bid}.self_attn.q_proj.weight"},
            "lm_head": {"gguf_name": "output.weight", "q4nx_name": "lm_head.weight"},
        }
        tensors = {f"blk.{i}.attn_q.weight" for i in range(4)}
        assert _warn_missing_config_entries(name_map, tensors) == ["output.weight"]

    def test_a_partially_present_layer_template_counts_as_present(self):
        # Any single match satisfies the template -- it is not "all {bid} present".
        name_map = {"q": {"gguf_name": "blk.{bid}.attn_q.weight",
                          "q4nx_name": "model.layers.{bid}.self_attn.q_proj.weight"}}
        tensors = {"blk.0.attn_q.weight"}          # one layer out of many
        assert _warn_missing_config_entries(name_map, tensors) == []

    def test_template_matching_nothing_is_reported(self):
        name_map = {"q": {"gguf_name": "blk.{bid}.attn_q.weight",
                          "q4nx_name": "model.layers.{bid}.self_attn.q_proj.weight"}}
        assert _warn_missing_config_entries(name_map, {"token_embd.weight"}) != []

    def test_placeholder_is_not_matched_literally(self):
        # Guards the regex escaping: '{bid}' must become (\d+), not match the
        # literal string "{bid}".
        name_map = {"q": {"gguf_name": "blk.{bid}.attn_q.weight", "q4nx_name": "x{bid}"}}
        assert _warn_missing_config_entries(name_map, {"blk.{bid}.attn_q.weight"}) != []


# --- config_coverage_report: which -f would have claimed this GGUF ---

class TestConfigCoverage:
    def test_a_full_match_outranks_a_partial_one(self):
        names = ["token_embd.weight", "output.weight", "blk.0.attn_q.weight"]
        ranked = config_coverage_report(names)
        assert ranked, "no config scored"
        assert ranked == sorted(ranked, key=lambda r: (r[3], r[1]), reverse=True)
        assert ranked[0][3] >= ranked[-1][3]

    def test_returns_frac_over_zero_when_total_is_zero(self):
        # A config with an empty name_map must not divide by zero.
        import json, tempfile, os
        with tempfile.TemporaryDirectory() as d:
            json.dump({"name_map": {}}, open(os.path.join(d, "empty.json"), "w"))
            json.dump({"name_map": {"e": {"gguf_name": "token_embd.weight",
                                          "q4nx_name": "model.embed_tokens.weight"}}},
                      open(os.path.join(d, "real.json"), "w"))
            ranked = config_coverage_report(["token_embd.weight"], configs_dir=d)
        assert all(r[2] > 0 for r in ranked), ranked

    def test_duplicate_name_maps_are_scored_once(self):
        # The five dense qwen3.5 configs share one name_map and differ only in
        # vision_config; the report must not list the same map five times.
        ranked = config_coverage_report(["token_embd.weight"])
        keys = [r[0] for r in ranked]
        assert len(keys) == len(set(keys))

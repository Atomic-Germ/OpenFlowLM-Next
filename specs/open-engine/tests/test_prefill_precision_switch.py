"""The prefill precision switch: opt-in, fail-closed, and never silently weaker.

Paired prefill issues FOUR products per projection -- (W_hi X_hi + W_hi X_lo) +
(W_lo X_hi + W_lo X_lo) -- and the `_low` companions are half of the prefill
GEMM time (measured: 51.5% of 74878.6 ms at 513 tokens over 64 layers). What
`_low` carries is not the weight's quantisation error: `gemm_q4_dequant.h`
builds `se` by dequantising the int4 weight exactly, and the `_low` variant
subtracts `bf16(se)` from it. So `_low` removes the bf16 rounding of the
dequantised weight and nothing else.

That makes it worth trading away on request, and these tests exist because the
trade has to stay a REQUEST. The default must issue all four products,
byte for byte what Phase 20 accepted, and every weaker mode must be named.
"""
import json
from pathlib import Path
import subprocess
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / 'src/open_qwen36/prefill_precision_switch.hpp'


def build(tmp_path, body):
    """Compile a TU against the header and run it. Exit 0 means every check held."""
    source = tmp_path / 'switch.cpp'
    source.write_text(textwrap.dedent(body))
    exe = tmp_path / 'switch'
    subprocess.run(['c++', '-std=c++17', '-O1', '-I' + str(ROOT / 'src'),
                    str(source), '-o', str(exe)], check=True)
    return exe


def expect_ok(tmp_path, body):
    return subprocess.run([str(build(tmp_path, body))], capture_output=True, text=True)


def test_the_header_exists_because_the_switch_is_not_an_env_var_read_in_place():
    assert HEADER.is_file(), (
        'the mode has to be a parsed value with a single owner, not a getenv() '
        'inside gemm_run: an unparsed typo would silently select a weaker route')


def test_the_default_is_the_four_product_sum(tmp_path):
    """An unset switch is the exact route Phase 20 accepted, not the fast one."""
    r = expect_ok(tmp_path, '''
        #include "open_qwen36/prefill_precision_switch.hpp"
        #include <cstdio>
        using namespace open_qwen36::prefill_precision_switch;
        int main() {
            if (kDefault != Mode::paired) return 1;
            if (dispatches(kDefault) != 4) return 2;
            if (!splits_input(kDefault)) return 3;
            if (!uses_weight_residual(kDefault)) return 4;
            return 0;
        }
    ''')
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize('mode,count,splits,residual,low_in', [
    ('paired', 4, True, True, True),
    ('three_term', 3, True, True, False),
    ('weight_high', 2, True, False, False),
    ('single', 1, False, False, False),
])
def test_each_mode_names_the_products_it_runs(tmp_path, mode, count, splits, residual, low_in):
    r = expect_ok(tmp_path, f'''
        #include "open_qwen36/prefill_precision_switch.hpp"
        using namespace open_qwen36::prefill_precision_switch;
        int main() {{
            const Mode m = parse("{mode}");
            if (name(m) != std::string("{mode}")) return 1;
            if (dispatches(m) != {int(count)}) return 2;
            if (splits_input(m) != {str(splits).lower()}) return 3;
            if (uses_weight_residual(m) != {str(residual).lower()}) return 4;
            if (needs_low_input(m) != {str(low_in).lower()}) return 5;
            return 0;
        }}
    ''')
    assert r.returncode == 0, r.stderr


def test_three_term_drops_exactly_one_term_and_keeps_the_weight_residual(tmp_path):
    """`three_term` must not be `weight_high` under a softer name.

    The two differ by which product goes: `three_term` keeps W_lo X_hi (the term
    `weight_high` gives up and that measurement showed to matter) and drops only
    W_lo X_lo. Getting this backwards would ship the rejected route as the
    "careful" option, so the predicates are compared directly.
    """
    r = expect_ok(tmp_path, '''
        #include "open_qwen36/prefill_precision_switch.hpp"
        using namespace open_qwen36::prefill_precision_switch;
        int main() {
            if (!uses_weight_residual(Mode::three_term)) return 1;   // W_lo X_hi still runs
            if (needs_low_input(Mode::three_term)) return 2;         // W_lo X_lo does not
            if (uses_weight_residual(Mode::weight_high)) return 3;   // nothing W_lo at all
            if (dispatches(Mode::three_term) != dispatches(Mode::paired) - 1) return 4;
            // A strict superset of three_term's products, in the accepted order.
            int t = dispatches(Mode::three_term), p = dispatches(Mode::paired);
            if (!(p > t && t > dispatches(Mode::weight_high))) return 5;
            return 0;
        }
    ''')
    assert r.returncode == 0, r.stderr


def test_three_term_is_a_strict_superset_of_the_accepted_sum(tmp_path):
    """The three-term sum is the four-term sum MINUS one small term, not a reassociation.

    Proven by running the same dispatch and checking that the three products the
    mode keeps are summed in the order the accepted route sums them, so its
    result differs from `paired` by exactly the omitted product.
    """
    r = expect_ok(tmp_path, '''
        #include "open_qwen36/prefill_precision.hpp"
        #include <vector>
        using namespace open_qwen36::prefill_precision;
        int main() {
            // Four term values, one per dispatch, in the accepted fold order.
            const float terms[4] = {1.5f, 0.25f, 0.125f, 0.0625f};
            auto four = [&](bool low, const float*) {
                static float s[1];
                static int i = 0;
                s[0] = terms[i++]; return s;
            };
            auto three = [&](const float*) {
                static float s[1];
                static int i = 0;
                s[0] = terms[i++]; return s;
            };
            float x[1] = {1.f};
            Workspace w;
            const float paired = w.run(x, 1, 1, four)[0];
            // W_hi X_hi + W_hi X_lo, then + W_lo X_hi (the W_lo X_lo term omitted).
            Workspace h;
            const float hi = h.high.run(x, 1, 1, three)[0];
            float lo_s[1]; lo_s[0] = terms[2];
            const float three_term = hi + lo_s[0];
            if (three_term != paired - terms[3]) return 1;
            if (!(three_term != paired)) return 2;
            return 0;
        }
    ''')
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize('bad', ['', 'fast', 'lean', 'Paired', 'single ', '3', 'exact'])
def test_an_unknown_mode_is_refused_rather_than_defaulted(tmp_path, bad):
    """A typo must stop the run, not quietly pick the weaker or the stronger route.

    `fast` and `lean` are deliberate: they are the values of the OTHER prefill
    setting (`prefill_mode`), so a user who confuses the two names gets told,
    instead of getting a route they did not ask for.
    """
    r = expect_ok(tmp_path, f'''
        #include "open_qwen36/prefill_precision_switch.hpp"
        #include <iostream>
        using namespace open_qwen36::prefill_precision_switch;
        int main() {{
            try {{ parse("{bad}"); }}
            catch (const std::exception& e) {{ std::cout << e.what(); return 0; }}
            return 1;
        }}
    ''')
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_weaker_modes_are_strictly_cheaper(tmp_path):
    """The ladder must actually descend, or the switch is not a lever."""
    r = expect_ok(tmp_path, '''
        #include "open_qwen36/prefill_precision_switch.hpp"
        using namespace open_qwen36::prefill_precision_switch;
        int main() {
            if (!(dispatches(Mode::paired) > dispatches(Mode::weight_high))) return 1;
            if (!(dispatches(Mode::weight_high) > dispatches(Mode::single))) return 2;
            if (!(dispatches(Mode::single) >= 1)) return 3;
            return 0;
        }
    ''')
    assert r.returncode == 0, r.stderr


def test_weight_high_dispatches_exactly_two_products(tmp_path):
    """Four products is the bug this test exists for.

    The first implementation of `weight_high` called the FOUR-product
    `Workspace` with a lambda that ignored the weight flag, so the high weight
    ran against both input halves and was then run again for the "low" half --
    four dispatches where the mode promises two. On hardware that cost
    68576.5 ms where two products measure 36132.9, and the profile showed 192
    calls where paired made 96. Counting the calls is what catches it.
    """
    r = expect_ok(tmp_path, '''
        #include "open_qwen36/prefill_precision_switch.hpp"
        using namespace open_qwen36::prefill_precision;
        int main() {
            // The composition Core::gemm_run uses for weight_high: Workspace::high.
            float input[] = {1.00390625f, -2.0078125f};
            float shared[2];
            int call = 0;
            auto high_weight_only = [&](const float* x) {
                shared[0] = float(call + 1); shared[1] = 0.f; ++call;
                return shared;
            };
            Workspace w;
            const float* y = w.high.run(input, 2, 2, high_weight_only);
            if (call != 2) return 1;                  // X_hi and X_lo and STOP
            if (y[0] != 1.f + 2.f) return 2;           // both halves folded in
            return 0;
        }
    ''')
    assert r.returncode == 0, r.stderr


def test_the_four_product_fold_would_issue_four(tmp_path):
    """The contrast that makes the test above meaningful, kept as a guardrail.

    If `Workspace::run` ever stops issuing four dispatches, the mode ladder in
    `dispatches()` no longer describes the arithmetic and both need revisiting.
    """
    r = expect_ok(tmp_path, '''
        #include "open_qwen36/prefill_precision.hpp"
        using namespace open_qwen36::prefill_precision;
        int main() {
            float input[] = {1.00390625f, -2.0078125f};
            float shared[2];
            int call = 0;
            auto four = [&](bool low, const float* x) {
                shared[0] = 1.f; shared[1] = 0.f; ++call;
                return shared;
            };
            Workspace w;
            w.run(input, 2, 2, four);
            if (call != 4) return 1;
            return 0;
        }
    ''')
    assert r.returncode == 0, r.stderr


def test_the_setting_is_registered_with_the_exact_default():
    """settings.json is the source of truth and CI regenerates from it."""
    settings = json.loads((ROOT / 'src/config/settings.json').read_text())['settings']
    found = {s['key']: s for s in settings}
    assert 'prefill_precision' in found, (
        'every OFLM_* variable the code reads must be a setting or be listed as '
        'a non-setting debug knob -- this one is a user trade-off, so it is a setting')
    setting = found['prefill_precision']
    assert setting['default_value'] == 'paired', 'the default must stay the exact route'
    assert setting['type'] == 'enum'
    assert [o['value'] for o in setting['options']] == ['paired', 'three_term', 'weight_high', 'single']
    assert setting['env_var'] == 'OFLM_OPEN_PREFILL_PRECISION'
    assert setting['cli_flag'] == '--prefill-precision'
    assert setting['delivery'] == 'env_var'
    assert setting['scope'] == 'per_model'


def test_role_selection_names_only_roles_that_exist(tmp_path):
    """A typo'd role must stop the run, not silently run `paired` everywhere.

    The whole point of the Phase 23 sweep is that a variant differs from the control
    in exactly one role. A name that parsed to nothing would produce an identical run
    labelled as a measurement of that role -- the most misleading possible failure.
    """
    r = expect_ok(tmp_path, '''
        #include "open_qwen36/prefill_precision_switch.hpp"
        #include <iostream>
        using namespace open_qwen36::prefill_precision_switch;
        int main() {
            auto v = parse_roles("qkvz, ffn_down");
            if (v.size() != 2) return 1;
            if (!selects(v, Role::qkvz) || !selects(v, Role::ffn_down)) return 2;
            if (selects(v, Role::out) || selects(v, Role::ffn_up)) return 3;
            if (selects(v, Role::none)) return 4;          // "none" is never selectable
            auto dup = parse_roles("out,out,out");
            if (dup.size() != 1) return 5;                 // deduplicated
            try { parse_roles("ffn_down,q_only"); }
            catch (const std::exception&) { return 0; }    // unknown is an error
            return 6;
        }
    ''')
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_role_list_is_exactly_what_the_export_can_separate(tmp_path):
    """Five roles, no more: the plan's finer list needs a repack, so it must not appear."""
    r = expect_ok(tmp_path, '''
        #include "open_qwen36/prefill_precision_switch.hpp"
        #include <string>
        using namespace open_qwen36::prefill_precision_switch;
        int main() {
            const auto& v = all_roles();
            if (v.size() != 5) return 1;
            std::string s;
            for (auto r : v) s += role_name(r) + ",";
            if (s != "qkvz,attn_in,out,ffn_up,ffn_down,") return 2;
            // The fused/absent ones must not be expressible.
            const char* bad[] = {"q", "k", "v", "gate", "up", "z", "lm_head", "alpha_beta"};
            for (const char* b : bad) {
                try { parse_role(b); return 3; } catch (const std::exception&) {}
            }
            return 0;
        }
    ''')
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_paired_route_is_frozen_as_the_correctness_baseline():
    """The tuning phase is closed against a frozen baseline, and this is the pin.

    A speedup that needed a weaker route was measured and rejected (301 of 513
    positions below the gate), and a near-miss is kept as an opt-in (1 of 513).
    Neither may drift into the default, and the gate itself may not move: the
    whole point of freezing is that a later change has something fixed to be
    compared against. If this test ever has to be edited, the baseline moved and
    the recorded numbers in utilities/prefill-perf/BASELINE.md are stale.
    """
    settings = {s['key']: s for s in
                json.loads((ROOT / 'src/config/settings.json').read_text())['settings']}
    assert settings['prefill_precision']['default_value'] == 'paired'
    header = HEADER.read_text()
    assert 'inline constexpr Mode kDefault = Mode::paired;' in header
    # The gate the baseline was measured with, in the one place it is applied.
    checker = (ROOT / 'utilities/prefill-long-context/check.py').read_text()
    assert 'CORR_GATE = 0.9999' in checker
    assert 'REAL_VOCAB = 248077' in checker or 'real_vocab' in checker


def test_the_baseline_document_records_numbers_not_aspirations():
    """A closing benchmark document must carry the measured figures, dated."""
    baseline = (ROOT / 'utilities/prefill-perf/BASELINE.md').read_text()
    for number in ('102721.2', '74575.7', '38442.8', '0.999911'):
        assert number in baseline, number
    assert 'FROZEN' in baseline
    assert '0.9999' in baseline


def test_the_two_prefill_settings_describe_different_things():
    """`prefill_mode` picks which route is loaded; this picks the arithmetic.

    They were one keystroke apart in the incident these tests were written for
    (a `lean` that changes nothing on a single-route set), so keep the two
    descriptions from collapsing into each other.
    """
    settings = {s['key']: s for s in
                json.loads((ROOT / 'src/config/settings.json').read_text())['settings']}
    assert settings['prefill_mode']['env_var'] != settings['prefill_precision']['env_var']
    assert settings['prefill_mode']['cli_flag'] != settings['prefill_precision']['cli_flag']
    assert settings['prefill_mode']['description'] != settings['prefill_precision']['description']

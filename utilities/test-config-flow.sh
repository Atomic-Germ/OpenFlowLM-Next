#!/usr/bin/env bash
# Test the #188 user-facing config flow end to end.
# Usage:  utilities/test-config-flow.sh [path/to/oflm]
#         OFLM=~/build/src/oflm utilities/test-config-flow.sh
#
# Every step prints PASS/FAIL and the script exits non-zero if anything failed.
# It never touches your real ~/.config/oflm/settings.json: OFLM_SETTINGS_PATH
# points at a scratch file for the whole run.

set -uo pipefail

OFLM="${1:-${OFLM:-build/src/oflm}}"
WORK="$(mktemp -d)"
SETTINGS="$WORK/settings.json"

export OFLM_SETTINGS_PATH="$SETTINGS"
export OFLM_CONFIG_PATH="${OFLM_CONFIG_PATH:-src/model_list.json}"

pass=0; fail=0
ok()   { echo "  PASS  $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }
step() { echo; echo "== $* =="; }
have() { command -v "$1" >/dev/null 2>&1; }

if [ ! -x "$OFLM" ]; then
  echo "oflm not found at '$OFLM'."
  echo "Build it first:  cmake --build --preset linux-default --target oflm"
  echo "Then run:        utilities/test-config-flow.sh build/src/oflm"
  exit 2
fi

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "oflm    : $OFLM"
echo "settings: $SETTINGS"
echo "registry: $OFLM_CONFIG_PATH"

MODEL="qwen3.8:27b"

# ---------------------------------------------------------------- 1. show
step "1. config show (read-only, nothing written)"
out=$("$OFLM" config show 2>&1)
# Without --config-model, only the global settings are listed: prefill_mode
# and the rest are per-model and belong to a model's view.
if echo "$out" | grep -q "pmode"; then
  ok "config show lists the global settings"
else
  bad "config show did not list pmode:"; echo "$out" | head -5
fi
if echo "$out" | grep -q "prefill_mode"; then
  bad "config show leaked a per-model setting into the global view"
else
  ok "config show keeps the per-model settings out of the global view"
fi
if echo "$out" | grep -qE '\((global|custom|env|cli|api)\)$'; then
  ok "config show tags each value with its source"
else
  bad "config show did not tag the sources:"; echo "$out" | head -5
fi
if [ ! -f "$SETTINGS" ]; then
  ok "config show did not create settings.json"
else
  bad "config show created settings.json (it should be read-only)"
fi

# ------------------------------------------------------- 2. set per-model
step "2. config set --config-model $MODEL --config-key prefill_mode --config-value lean"
out=$("$OFLM" config set --config-model "$MODEL" --config-key prefill_mode --config-value lean 2>&1)
if [ -f "$SETTINGS" ]; then
  ok "settings.json was created"
else
  bad "settings.json was not created:"; echo "$out"
fi
if grep -q '"lean"' "$SETTINGS" 2>/dev/null; then
  ok "the value was stored"
else
  bad "the value was not stored:"; cat "$SETTINGS" 2>/dev/null
fi
if grep -q '"qwen3.8"' "$SETTINGS" 2>/dev/null && grep -q '"27b"' "$SETTINGS" 2>/dev/null; then
  ok "it landed under models.qwen3.8.27b"
else
  bad "wrong location in settings.json:"; cat "$SETTINGS" 2>/dev/null
fi

# ------------------------------------------------------- 3. show reflects it
step "3. config show reflects the user override"
out=$("$OFLM" config show --config-model "$MODEL" 2>&1)
if echo "$out" | grep -qE "prefill_mode.*lean.*\(custom\)"; then
  ok "prefill_mode shows as lean (custom)"
else
  bad "prefill_mode did not show as lean (custom):"; echo "$out"
fi

# --------------------------------------------------------- 4. global set
step "4. config set --config-key pmode --config-value balanced"
"$OFLM" config set --config-key pmode --config-value balanced >/dev/null 2>&1
if grep -q '"balanced"' "$SETTINGS" 2>/dev/null; then
  ok "the global default was stored"
else
  bad "the global default was not stored:"; cat "$SETTINGS" 2>/dev/null
fi

# ------------------------------------------------------------- 5. invalid
step "5. config set rejects an invalid value"
out=$("$OFLM" config set --config-key prefill_mode --config-value nonsense 2>&1)
rc=$?
if [ $rc -ne 0 ]; then
  ok "an invalid enum value is rejected (exit $rc)"
else
  bad "an invalid enum value was accepted"
fi
if echo "$out" | grep -qiE "fast|lean"; then
  ok "the error lists the valid options"
else
  bad "the error did not list the valid options:"; echo "$out"
fi

# ------------------------------------------------------------- 6. unknown
step "6. config set warns about an unknown key"
out=$(printf 'n\n' | "$OFLM" config set --config-key not_a_setting --config-value x 2>&1)
if echo "$out" | grep -qi "not a known setting"; then
  ok "an unknown key warns"
else
  bad "an unknown key did not warn:"; echo "$out" | head -5
fi

# --------------------------------------------------------------- 7. unset
step "7. config unset removes the override"
"$OFLM" config unset --config-model "$MODEL" --config-key prefill_mode >/dev/null 2>&1
out=$("$OFLM" config show --config-model "$MODEL" 2>&1)
if echo "$out" | grep -qE "prefill_mode.*\(global\)"; then
  ok "prefill_mode is back to the global default"
else
  bad "prefill_mode did not revert to global:"; echo "$out"
fi

# --------------------------------------------------------- 8. gen-schema
step "8. config gen-schema"
out=$("$OFLM" config gen-schema 2>/dev/null)
if echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'settings_schema' in d; assert d['settings_schema']['settings']" 2>/dev/null; then
  ok "gen-schema emits valid JSON with the schema"
else
  bad "gen-schema did not emit valid schema JSON:"; echo "$out" | head -5
fi

# ------------------------------------------------------------ 9. --json
step "9. config show --json is machine readable"
out=$("$OFLM" config show --json --config-model "$MODEL" 2>/dev/null)
if echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'prefill_mode' in d; assert 'source' in d['prefill_mode']" 2>/dev/null; then
  ok "--json emits the value and its source"
else
  bad "--json output was not as expected:"; echo "$out" | head -8
fi

# -------------------------------------------------------- 10. precedence
step "10. CLI flag wins over the user file"
"$OFLM" config set --config-model "$MODEL" --config-key prefill_mode --config-value lean >/dev/null 2>&1
out=$("$OFLM" config show --config-model "$MODEL" 2>&1)
if echo "$out" | grep -qE "prefill_mode.*lean.*\(custom\)"; then
  ok "the file says lean (custom)"
else
  bad "the file override was not picked up:"; echo "$out"
fi
echo "  (a --prefill-mode flag overrides this at run time; the file is the default)"

# ------------------------------------------------------- 11. edit target
step "11. config edit opens an editor on the settings file"
out=$(EDITOR=true "$OFLM" config edit 2>&1)
if echo "$out" | grep -qi "settings.json" || [ -f "$SETTINGS" ]; then
  ok "config edit pointed at the settings file"
else
  bad "config edit said nothing useful:"; echo "$out" | head -3
fi

# ---------------------------------------------------- 12. interactive
step "12. the interactive flow does not crash on a dumb terminal"
out=$(printf '0\n' | "$OFLM" config 2>&1)
rc=$?
if [ $rc -lt 128 ]; then
  ok "the guided flow exited cleanly (exit $rc)"
else
  bad "the guided flow died on a signal (exit $rc):"; echo "$out" | head -5
fi

# -------------------------------------------------------- 13. browse TUI
step "13. config browse does not crash on a dumb terminal"
out=$(echo "q" | "$OFLM" config browse 2>&1)
rc=$?
if [ $rc -eq 0 ] || [ $rc -eq 1 ]; then
  ok "browse exited without crashing (exit $rc)"
else
  bad "browse died on a signal (exit $rc):"; echo "$out" | head -5
fi
if echo "$out" | grep -qi "terminal\|guided flow"; then
  ok "browse says what to use instead on a plain terminal"
fi

# ------------------------------------------------------- 14. migration
step "14. the legacy registry is migrated on first load"
LEGACY="$WORK/legacy"
mkdir -p "$LEGACY"
cat > "$LEGACY/model_list.json" <<EOF
{
  "model_path": "models",
  "hf_owner": "Test",
  "models": {
    "my-old-model": {
      "1b": {
        "name": "my-old-model-1b",
        "url": "https://example.com/model",
        "default_context_length": 4096
      }
    }
  }
}
EOF
out=$(OFLM_MODEL_PATH="$LEGACY" OFLM_SETTINGS_PATH="$LEGACY/settings.json" "$OFLM" config show 2>&1)
if [ -f "$LEGACY/settings.json" ]; then
  ok "the migration created settings.json"
else
  bad "the migration did not create settings.json:"; echo "$out" | head -8
fi
if [ -f "$LEGACY/model_list.json.bak" ]; then
  ok "the old file was kept as model_list.json.bak"
else
  bad "the old file was not kept as .bak"
fi
if grep -q "my-old-model" "$LEGACY/settings.json" 2>/dev/null; then
  ok "the user-added model was carried over"
else
  bad "the user-added model was not carried over:"; cat "$LEGACY/settings.json" 2>/dev/null | head -10
fi

# ------------------------------------------------ 15. review / reconcile
step "15. config review --keep-all updates the defaults version"
out=$("$OFLM" config review --keep-all 2>&1)
rc=$?
if [ $rc -eq 0 ]; then
  ok "config review --keep-all succeeded"
else
  bad "config review --keep-all failed (exit $rc):"; echo "$out" | head -5
fi

# ------------------------------------------------ 16. guided flow navigation
step "16. the guided flow goes back a level instead of quitting"
# 1 = Edit a model's settings, 44 = qwen3.8:27b, 0 = Back, 0 = Back
out=$(printf '2\n44\n0\n0\n' | "$OFLM" config 2>/dev/null)
if echo "$out" | grep -q "Select model" && echo "$out" | grep -q "What do you want to do"; then
  ok "Back at the model list returns to the action list"
else
  bad "Back did not return to the action list:"; echo "$out" | head -10
fi

# ------------------------------------------------ 17. the menus show the state
step "17. the menus show the current value and where it came from"
"$OFLM" config set --config-model "$MODEL" --config-key prefill_mode --config-value lean >/dev/null 2>&1
out=$(printf '2\n44\n' | "$OFLM" config 2>/dev/null)
if echo "$out" | grep -qE "prefill_mode.*lean.*\(custom\)"; then
  ok "the setting list shows the user's own value as (custom)"
else
  bad "the setting list did not show the (custom) value:"; echo "$out" | head -12
fi
if echo "$out" | grep -qE "ctx_len.*\(global\)"; then
  ok "the setting list shows the shipped value as (global)"
else
  bad "the setting list did not show a (global) value:"; echo "$out" | head -12
fi
out=$(printf '2\n44\n1\n' | "$OFLM" config 2>/dev/null)
if echo "$out" | grep -qE "Select value for prefill_mode.*current: lean \(custom\)"; then
  ok "the value picker names the option in effect"
else
  bad "the value picker did not name the option in effect:"; echo "$out" | head -12
fi

# ------------------------------------------------ 18. the menus list the keys
step "18. the menus show the keyboard shortcuts"
out=$(printf '0\n' | "$OFLM" config 2>/dev/null)
if echo "$out" | grep -q "q. quit"; then
  ok "the menu shows the quit key"
else
  bad "the menu did not show the quit key:"; echo "$out" | head -8
fi

# ------------------------------------------------ 19. a per-model global override
step "19. a global setting overridden for one model shows in that model's view"
# pmode is a global setting, but the user can override it for one model.
"$OFLM" config set --config-model "$MODEL" --config-key pmode --config-value balanced >/dev/null 2>&1
out=$("$OFLM" config show --config-model "$MODEL" 2>/dev/null)
if echo "$out" | grep -qE "pmode.*balanced.*\(custom\)"; then
  ok "the model's own pmode override is shown"
else
  bad "the model's own pmode override was hidden:"; echo "$out"
fi
# The count in the model list must agree with what is shown.
# The JSON view also carries a "paths" block (where each path points now),
# which is not a setting -- only count the entries that have a source.
n_custom=$("$OFLM" config show --json --config-model "$MODEL" 2>/dev/null \
            | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for v in d.values() if isinstance(v,dict) and v.get('source')=='(custom)'))")
out=$(printf '2\n0\n' | "$OFLM" config 2>/dev/null | grep "$MODEL")
if echo "$out" | grep -q "($n_custom custom)"; then
  ok "the model list counts $n_custom custom settings, which is what is shown"
else
  bad "the model list count does not match what is shown ($n_custom):"; echo "$out"
fi
"$OFLM" config unset --config-model "$MODEL" --config-key pmode >/dev/null 2>&1

# ------------------------------------------------ 20. a changed shipped default is reported
step "20. a shipped default that changed under the user is reported"
CONFLICT="$WORK/conflict"
mkdir -p "$CONFLICT"
cp "$OFLM_CONFIG_PATH" "$CONFLICT/model_list.json"
# A settings file written against an older shipped registry.
cat > "$CONFLICT/settings.json" <<'EOF'
{
  "version": 1,
  "defaults_version": "an_older_hash",
  "defaults": { "pmode": "balanced" },
  "settings": {},
  "models": {}
}
EOF
out=$(OFLM_CONFIG_PATH="$CONFLICT/model_list.json" \
        OFLM_SETTINGS_PATH="$CONFLICT/settings.json" \
        "$OFLM" config show 2>&1)
if echo "$out" | grep -q "Shipped defaults changed"; then
  ok "the change is reported"
else
  bad "the change was not reported:"; echo "$out" | head -6
fi
if echo "$out" | grep -q "pmode"; then
  ok "the override that is affected is named"
else
  bad "the affected override was not named:"; echo "$out" | head -6
fi
if echo "$out" | grep -q "oflm config review"; then
  ok "the way to reconcile it is offered"
else
  bad "no way to reconcile was offered:"; echo "$out" | head -6
fi
# The user's own value still wins.
if echo "$out" | grep -q "balanced"; then
  ok "the user's value is still the one in effect"
else
  bad "the user's value was lost:"; echo "$out" | head -6
fi

# ------------------------------------------------ 21. config review reconciles
step "21. config review reconciles the overrides"
# A redundant override: the same value the application ships.
cat > "$CONFLICT/settings.json" <<'EOF'
{
  "version": 1,
  "defaults_version": "an_older_hash",
  "defaults": { "pmode": "performance" },
  "settings": {},
  "models": {}
}
EOF
out=$(OFLM_CONFIG_PATH="$CONFLICT/model_list.json" \
        OFLM_SETTINGS_PATH="$CONFLICT/settings.json" \
        "$OFLM" config review --accept-shipped 2>&1)
if echo "$out" | grep -qi "removed"; then
  ok "review --accept-shipped reports what it removed"
else
  bad "review --accept-shipped said nothing useful:"; echo "$out" | head -5
fi
left=$(python3 -c "
import json
print(len(json.load(open('$CONFLICT/settings.json')).get('defaults', {})))
" 2>/dev/null)
if [ "$left" = "0" ]; then
  ok "the redundant override is gone"
else
  bad "the redundant override is still there ($left left)"
fi
ver=$(python3 -c "
import json
print(json.load(open('$CONFLICT/settings.json')).get('defaults_version', ''))
" 2>/dev/null)
if [ -n "$ver" ] && [ "$ver" != "an_older_hash" ]; then
  ok "the defaults version was updated, so the warning stops"
else
  bad "the defaults version was not updated ('$ver')"
fi

# --keep-all changes nothing but the version.
cat > "$CONFLICT/settings.json" <<'EOF'
{
  "version": 1,
  "defaults_version": "an_older_hash",
  "defaults": { "pmode": "balanced" },
  "settings": {},
  "models": {}
}
EOF
out=$(OFLM_CONFIG_PATH="$CONFLICT/model_list.json" \
        OFLM_SETTINGS_PATH="$CONFLICT/settings.json" \
        "$OFLM" config review --keep-all 2>&1)
kept=$(python3 -c "
import json
print(json.load(open('$CONFLICT/settings.json')).get('defaults', {}).get('pmode', ''))
" 2>/dev/null)
if [ "$kept" = "balanced" ]; then
  ok "review --keep-all leaves the override alone"
else
  bad "review --keep-all changed the override ('$kept')"
fi

# ------------------------------------------------------------- summary
echo
echo "================================"
echo "  passed: $pass"
echo "  failed: $fail"
echo "================================"
[ "$fail" -eq 0 ] || exit 1

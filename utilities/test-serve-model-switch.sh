#!/usr/bin/env bash
# Verify on a real NPU that `oflm serve` gives each model its OWN settings.
#
# The bug this guards against: the settings were resolved once, for the model
# named on the command line, so every model a server later loaded got the
# first model's context length and prefill mode.
#
# Usage:  utilities/test-serve-model-switch.sh [path/to/oflm]
#
# Needs: an NPU, and two chat models with real weights. The script finds them
# in the model list and skips the ones that are not on disk.

set -uo pipefail

OFLM="${1:-build/src/oflm}"
WORK="$(mktemp -d)"

pass=0; fail=0
ok()  { echo "  PASS  $*"; pass=$((pass+1)); }
bad() { echo "  FAIL  $*"; fail=$((fail+1)); }

if [ ! -x "$OFLM" ]; then
  echo "oflm not found at '$OFLM'."
  echo "Build it first:  cmake --build --preset linux-default --target oflm"
  exit 2
fi

# The model directory is <models_dir>/<model_path>/<name>, and the model
# list says what model_path is. The user sets OFLM_MODEL_PATH to the dir
# that holds "models/".
: "${OFLM_MODEL_PATH:?set OFLM_MODEL_PATH to the directory that holds the models/}"
REGISTRY="${OFLM_CONFIG_PATH:-src/model_list.json}"
export OFLM_CONFIG_PATH="$REGISTRY"
export OFLM_SETTINGS_PATH="$WORK/settings.json"
export OFLM_XCLBIN_PATH="${OFLM_XCLBIN_PATH:-$(dirname "$REGISTRY")}"

# Pick two chat models that are actually on disk.
read -r MODEL_A MODEL_B <<EOF
$(python3 - "$REGISTRY" "$OFLM_MODEL_PATH" <<'PY'
import json, os, sys
reg, models_dir = sys.argv[1], sys.argv[2]
d = json.load(open(reg))
sub = d.get("model_path", "models")
found = []
for t, sizes in d.get("models", {}).items():
    for s, e in sizes.items():
        name = e.get("name", "")
        p = os.path.join(models_dir, sub, name)
        w = os.path.join(p, "model.q4nx")
        # A real weight file, not a git-lfs pointer.
        if os.path.isfile(w) and os.path.getsize(w) > 1_000_000:
            found.append(f"{t}:{s}")
print(" ".join(found[:2]))
PY
)
EOF

if [ -z "${MODEL_A:-}" ] || [ -z "${MODEL_B:-}" ]; then
  echo "Need two chat models with real weights under $OFLM_MODEL_PATH."
  echo "Found none or only one. Skipping."
  exit 3
fi

echo "models: $MODEL_A and $MODEL_B"
echo "registry: $REGISTRY"
echo "settings: $OFLM_SETTINGS_PATH"

# Give each model a setting of its own, so a leak is visible.
python3 - "$MODEL_A" "$MODEL_B" <<'PY'
import json, os, sys
a, b = sys.argv[1].split(":")[0], sys.argv[2].split(":")[0]
sa, sb = sys.argv[1].split(":")[1], sys.argv[2].split(":")[1]
settings = {
    "version": 1,
    "defaults_version": "test",
    "defaults": {},
    "settings": {
        a: {sa: {"prefill_mode": "lean", "ctx_len": 8192}},
        b: {sb: {"prefill_mode": "fast", "ctx_len": 16384}},
    },
    "models": {},
}
path = os.environ["OFLM_SETTINGS_PATH"]
with open(path, "w") as f:
    json.dump(settings, f, indent=2)
print("wrote", path)
PY

PORT=${PORT:-52699}
LOG="$WORK/serve.log"
cleanup() {
  [ -n "${SRV_PID:-}" ] && kill -TERM "$SRV_PID" 2>/dev/null
  sleep 2
  rm -rf "$WORK"
}
trap cleanup EXIT

echo
echo "== starting the server on $MODEL_A =="
"$OFLM" serve "$MODEL_A" --port "$PORT" > "$LOG" 2>&1 &
SRV_PID=$!

# Wait for the port to answer.
for i in $(seq 1 120); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1; then break; fi
  sleep 1
done

if ! curl -s -m 5 "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1; then
  echo "  the server did not come up:"
  tail -20 "$LOG"
  exit 1
fi

# A request for the second model makes the server load it.
echo "== asking for $MODEL_B =="
curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_B\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" \
  > "$WORK/b.json" 2>&1

if grep -q '"error"' "$WORK/b.json"; then
  bad "the request for $MODEL_B failed:"
  head -c 400 "$WORK/b.json"; echo
  tail -20 "$LOG"
  exit 1
fi
ok "the request for $MODEL_B was served"

# And back to the first one.
echo "== asking for $MODEL_A again =="
curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_A\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" \
  > "$WORK/a.json" 2>&1

if grep -q '"error"' "$WORK/a.json"; then
  bad "the request for $MODEL_A failed:"
  head -c 400 "$WORK/a.json"; echo
  tail -20 "$LOG"
  exit 1
fi
ok "the request for $MODEL_A was served"

echo
echo "== the settings in effect for each load =="
grep "settings for" "$LOG"

# Each model must have its own values.
line_a=$(grep "settings for '$MODEL_A'" "$LOG" | tail -1)
line_b=$(grep "settings for '$MODEL_B'" "$LOG" | tail -1)

if echo "$line_a" | grep -q "prefill_mode=lean" && echo "$line_a" | grep -q "ctx_len=8192"; then
  ok "$MODEL_A has its own settings"
else
  bad "$MODEL_A does not have its own settings:"; echo "    $line_a"
fi

if echo "$line_b" | grep -q "prefill_mode=fast" && echo "$line_b" | grep -q "ctx_len=16384"; then
  ok "$MODEL_B has its own settings"
else
  bad "$MODEL_B does not have its own settings:"; echo "    $line_b"
fi

echo
echo "================================"
echo "  passed: $pass"
echo "  failed: $fail"
echo "================================"
[ "$fail" -eq 0 ] || exit 1

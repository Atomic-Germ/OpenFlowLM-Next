#!/usr/bin/env bash
#
# Reproduce the open engine's intermittent kernel timeout, unattended.
#
# The symptom: a dx dispatch that normally takes about 2 ms sits for the whole
# of OFLM_OPEN_TIMEOUT_MS (60 s by default), Engine::guarded poisons the engine
# and rebuilds it, and the request dies. Since 2026-09-13 the engine PRINTS what
# failed, so the server log is the authoritative record:
#
#   open_qwen36: request failed: open_qwen36: kernel dx at position 613 ended in
#   ERT state 8 (timeout)
#
# Two things this is built around, both learned the hard way:
#
#  * It is NOT tied to the turn boundary. It has hit after a 36-token first-round
#    prefill and after a 17-token cache-reusing one in the same run. What those
#    have in common is the first decode dispatch after a prefill.
#  * A failed streamed request sends the client NOTHING - the error body is
#    written only for a deferred request - so the client sits until its own
#    timeout. That is why the client gets a deadline here, and why failure is
#    detected from the server log rather than from the client exit code.
#
# Usage:  utilities/repro-open-timeout.sh [attempts] [tag] [kernel subdir]
#
#   utilities/repro-open-timeout.sh 12
#   utilities/repro-open-timeout.sh 12 qwen2.5-it:3b open_kernels_slow
#
# The slow kernel set is still beside the model as open_kernels_slow, so the
# third argument is how you compare rates between the two attention paths.
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
N=${1:-12}
TAG=${2:-qwen2.5-it:3b}
KERNELS=${3:-}
CLIENT_DEADLINE=${CLIENT_DEADLINE:-180}
OUT=${OUT:-$REPO/.repro-open-timeout}
MODEL_DIR_NAME=${MODEL_DIR_NAME:-Qwen2.5-3B-Instruct-NPU2}

mkdir -p "$OUT"
: > "$OUT/summary.txt"
say() { echo "$@" | tee -a "$OUT/summary.txt"; }

export OFLM_CONFIG_PATH="${OFLM_CONFIG_PATH:-$HOME/.config/oflm/model_list.json}"
export OFLM_XCLBIN_PATH="${OFLM_XCLBIN_PATH:-$HOME/.config/oflm}"
export OFLM_MODEL_PATH="${OFLM_MODEL_PATH:-$HOME/.flm}"
export OFLM_QWEN2_ENGINE=open
if [ -n "$KERNELS" ]; then
    export OFLM_OPEN_KERNELS_DIR="$OFLM_MODEL_PATH/models/$MODEL_DIR_NAME/$KERNELS"
fi

cat > "$OUT/client.py" <<'PY'
import sys
sys.path.insert(0, ".")
from oflm_test import main
sys.argv = ["oflm-test", "--llm", "--model", sys.argv[1]]
main()
PY

say "tag $TAG   attempts $N   kernels ${KERNELS:-beside the model}"
say "started $(date)"

"$REPO/src/build/oflm.exe" serve "$TAG" > "$OUT/serve.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

ready=0
for _ in $(seq 1 120); do
    if curl -s -m 2 http://127.0.0.1:52625/v1/models > /dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 $SRV 2>/dev/null; then say "server died before it was ready"; cat "$OUT/serve.log"; exit 1; fi
    sleep 2
done
[ "$ready" = 1 ] || { say "server never came up"; exit 1; }
say "engine: $(grep -i 'open kernels' "$OUT/serve.log" | head -1)"

fails=0
for i in $(seq 1 "$N"); do
    before=$(grep -c "request failed" "$OUT/serve.log" 2>/dev/null || true)
    t0=$(date +%s)
    ( cd "$REPO/utilities/oflm-test" && timeout "$CLIENT_DEADLINE" \
        env PYTHONIOENCODING=utf-8 python "$OUT/client.py" "$TAG" ) > "$OUT/run_$i.log" 2>&1
    t1=$(date +%s)
    after=$(grep -c "request failed" "$OUT/serve.log" 2>/dev/null || true)
    if [ "${after:-0}" -gt "${before:-0}" ]; then
        fails=$((fails + 1))
        say "attempt $i: FAIL after $((t1 - t0))s"
        grep "request failed" "$OUT/serve.log" | tail -n 1 | sed 's/^/    /' | tee -a "$OUT/summary.txt"
    else
        say "attempt $i: ok   ($((t1 - t0))s)"
    fi
done

say "----"
say "$fails of $N attempts hit it"
say "failing dispatches:"
grep -o "kernel [a-z_]* at position [0-9]*" "$OUT/serve.log" | sort | uniq -c | sed 's/^/    /' \
    | tee -a "$OUT/summary.txt"
say "finished $(date)"

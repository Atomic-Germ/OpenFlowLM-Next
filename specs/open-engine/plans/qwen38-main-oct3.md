# Main integration after shared WideDeltaNet merge

2026-10-03: integrate origin/main `0ceb46d` (includes upstream/main `b16e6ab`)
into the precision branch at `b8a51bb`. Upstream PR121 is our earlier shared
WideDeltaNet stage, so its overlap must not discard later segmented FFN,
streamed inputs, compensated recurrence/conv, or diagnostic modes.

Other changes add Docker builds, CLI fixes, correct DX_STOP dispatch ordering,
and keep H2560 band loops rolled to fit the 4B program memory limit. The latter
uses a measured catalogue entry only for qwen35/H2560. It does not change H5120
arithmetic or close the real27B accuracy gate.

Merge resolution preserves the later precision code and combines the upstream
band helper with compact and ordinary up/gate traversal. Initial regression
run: seven failures (six simulation NameErrors and the generated-source rule).
The simulation now loads band_range. The hand-written activation probe moves
from `.cc` to `.cpp`, outside the generated-TU suffix; its build inputs follow
the rename. Subsequent CPU suite: **854 passed, 47 skipped**.

C++ runtime builds with the installed matching XRT SDK and passes all three
CTest cases in `/tmp/oflm-main-oct3`. `bash -n build_in_docker.sh` passes; this
check does not establish a fresh Docker image build or live CLI/server quality.

The activation-series FFN rebuild in `build_main_oct3/ffn` passes all26
primitive checks; all13 arenas are byte-identical to
`build_activation749/compact/ffn`. Full replay completes3564 NPU dispatches.
The original weights, references and thresholds are unchanged.

See [.opencode/skill/qwen38-wide-main-integration](../../../.opencode/skill/qwen38-wide-main-integration/SKILL.md)
for reproduction. Logs: `/tmp/main-oct3-*.log`.

All8190 full-model binary captures are byte-identical to
`build_activation749/full`. Acceptance remains18891/18896 slice and4121/4122
decode: the same six numerical failures, with no new regression from main.
This merge does not establish production27B support; PR4 remains incomplete.

Rebuilt FFN hashes:
- xclbin: `713d6d786deadf1813ad4f3cec9512831fa0d20475125b132171a563d8c0b316`
- instructions: `e8cc08c29bbca5afbdfe59493baf6d4655e6853f0a4c369b25847e91337ce708`

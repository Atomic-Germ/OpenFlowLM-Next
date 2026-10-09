# Plan: `oflm pull` upgrades an installed model

**Status:** done (2026-10-08); requirements merged into `specs/pull/spec.md`.
**Spec:** new, `specs/pull/spec.md` (prefix `PULL`). No spec owns `oflm pull` today.
**Branch:** `fix/pull-upgrade`, from `origin/main`, its own PR. #137 keeps its "Known issue" note
until this merges.

## Problem

Measured 2026-10-08 while upgrading `flux2-klein:4b` from #136's install (HF `main`) to #137's
pinned revision `d84e3fd6`. Nine of the model's 21 files differ:

| file | change | what `oflm pull` did |
|---|---|---|
| `bundle.json`, `config.json`, `schedule_512.json`, `schedule_1024.json` | size | warned "treating it as missing", then kept the old file |
| `vae_W.bin` | content, same size | nothing: no check can see it |
| two edit schedules, `vae_enc_S.bin`, `vae_enc_W.bin` | new | downloaded |

The pull reported success. The result is a mix of old and new files. The engine refuses it on the
layout hash. A stale `vae_W.bin` on its own would have given silently wrong images.

`oflm pull --force` downloaded nothing either.

This is not specific to klein. `oflm pull` downloads every registry model, 46 on main, 11 of them
pinned to a revision. The next revision of any of them that changes a file hits the same path.

## Causes (`src/pull/model_downloader.cpp`, `download_model.cpp`)

1. **Two predicates.** `get_missing_files` reports a file that is absent or the wrong size.
   `build_download_list` fetches only files that are absent. Status and download disagree.
2. **No record of what was installed.** Size is a file's only identity. Nothing says which registry
   entry a file was downloaded for, so a same-size change is invisible.
3. **`--force` relies on a deletion that no longer happens.** It calls `verify_and_clean_files`,
   which stopped deleting mismatched files when hash checks became advisory. `build_download_list`
   then skips every present file, so `--force` never downloads anything, in any status.
4. **Downloads write in place.** `download_file` opens the target with `"wb"` and removes it on a
   failure. Today that only ever touches an absent or truncated file. Once present files are
   re-downloaded, an interrupted pull would destroy a working file.

## Design

- **One predicate.** `needs_fetch(registry entry, local state)` decides for each file, and both
  status (`get_missing_files`) and `build_download_list` use it. A file needs fetching when:
  - it is absent;
  - its size differs from the registry's; or
  - the install record names a different oid than the registry.
- **An install record.** `<model dir>/.oflm-files.json` maps each file's path to the registry oid it
  was downloaded *for*. It is written after each file completes.
  - It records the registry's oid, not a hash computed locally. A repo that serves bytes the
    registry's oid does not describe (LFS vs git-blob, mirrors) is then fetched once and recorded,
    not fetched forever. That loop is the reason hash checks became advisory, and this keeps them
    advisory.
  - `oflm remove` already deletes every regular file in the directory, the record included.
- **Installs with no record.** That is every install today.
  - Status uses size only, as now, so `oflm list` and every model load stay as cheap as they are.
  - `oflm pull`, explicit or automatic, hashes each present file once against the registry's oid
    (sha256 for LFS, git-blob otherwise, as `verify_and_clean_files` does). A match is recorded
    without a download; a mismatch is downloaded. After that, the install has a record.
  - An explicit `oflm pull <tag>` on a model with no record runs this once even when the status is
    Ready. That is how a legacy install whose only change kept its size gets fixed.
- **`--force`** fetches every file, whatever the status.
- **Atomic replacement.** A download goes to `<file>.part` and is renamed over the old file only
  when it completes. That needs room for the largest file twice, briefly: 7.5 GB for klein's
  `weights.bin`.

Unchanged: the registry formats (`model_list.json`, `model_info.json`), the version check
(`check_model_compatibility`), and hash mismatches staying advisory.

## Cost

- Status checks (`oflm list`, every load): a stat per file plus one small JSON read, as now.
- The first pull of an install with no record reads every file once, to hash it. For klein that is
  9 GB. The implementation measures the actual time and records it in the spec.

## Requirements (new, `specs/pull/spec.md`)

| ID | Statement | Verification |
|---|---|---|
| PULL-STALE | A file that is absent, the wrong size, or recorded for a different registry oid is reported missing by status and fetched by `pull`. A file that matches is neither. | test |
| PULL-RECORD | After a pull, the record names each file's registry oid. A pull of an install with no record hashes the present files once: matches are recorded without a download, mismatches are downloaded. | test |
| PULL-FORCE | `oflm pull --force` fetches every file. | test |
| PULL-ATOMIC | An interrupted or failed download leaves the previous file in place. | manual |
| PULL-UPGRADE | #136's klein install upgrades to `d84e3fd6` with one `oflm pull`, with no `remove` and no `--force`. | manual |

**Tests.** The decision is a pure function of the registry entries, the file sizes and the record,
so a CTest (`src/pull/pull_plan_test.cpp`, traced to PULL-STALE, -RECORD and -FORCE) calls it with
real inputs. The one-time hash runs on files the test writes to a temp directory. Neither needs
the network.

**Manual procedures**, to be written into the spec:
- PULL-ATOMIC: interrupt a re-download, then check that the old file is still in place and the
  next pull completes.
- PULL-UPGRADE: on a copy of #136's install, run the upgrade with `OFLM_CONFIG_PATH` pointed at
  #137's registry. Expected: the 5 changed files and the 4 new ones are downloaded, the other 12
  are kept, and the model loads and generates.

## Not in scope

- An Outdated status (version check) makes `is_model_downloaded` hash every file on every load,
  through `verify_and_clean_files`. That is slow, but it is unrelated to this bug.
- Deleting files the registry no longer lists.
- GGUF: `oflm pull` downloads none today. GGUF is only an input to `oflm pack`.

## Open question (answered 2026-10-08: accepted)

1. **A legacy install whose only change kept its size.** `oflm list` and the automatic pull won't
   notice it until someone runs `oflm pull <tag>`. **Recommended: accept that**, so status stays
   cheap. The alternative is to hash during status for installs with no record, which makes the
   first `oflm list` after the upgrade read every installed model in full.

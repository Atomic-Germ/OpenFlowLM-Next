# `oflm pull`: installing and upgrading a model

How `oflm pull` decides which of a model's files to download, for an explicit pull, `--force`, and
the automatic pull `run`, `serve`, `image` and `bench-embed` make when a model is missing or
outdated. The code is `src/pull/`: `model_downloader.cpp` drives a pull, `install_record.cpp`
decides what it fetches, and `download_model.cpp` downloads one file.

Directory name gives the prefix: `PULL`.

Two registry files describe a model. `model_list.json` names its files, and `model_info.json`
gives each file's size and oid (an LFS file's sha256, otherwise its git blob oid), copied from the
repository at the revision the entry pins. An upgrade is a new revision of those two files.

Hash checks of downloaded bytes against the registry stay advisory. A repository can serve bytes
the registry's oid does not describe (LFS vs git-blob, mirrors), and a failing check must not
block a pull or loop on re-downloads. The install record below compares registry with registry,
which needs no hash, so it does not reopen that.

## Requirements

### PULL-STALE: a file that is not the registry's is fetched, and status says so
**Applies to:** openflowlm-next (`src/pull/install_record.cpp`, `src/pull/model_downloader.cpp`)
**Verification:** test
**Test:** `src/pull/pull_plan_test.cpp` (CTest `pull_plan`, no network)

A file needs fetching when it is absent, when its size differs from the registry's, or when the
install record names a different oid than the registry. The status check (`oflm list`, every model
load) and the pull use that one test, so a file that status calls missing is a file the pull
downloads. The status check never reads a file's contents.

**Acceptance criteria:**
- An absent file, a file of the wrong size, and a file recorded for another oid each need
  fetching; a file of the right size that is unrecorded, or recorded for the registry's oid,
  does not.
- A registry oid that changes fetches a file the record names for the old one.

### PULL-RECORD: what was installed is recorded, and an install with no record is hashed once
**Applies to:** openflowlm-next (`src/pull/install_record.cpp`, `src/pull/model_downloader.cpp`)
**Verification:** test
**Test:** `src/pull/pull_plan_test.cpp` (CTest `pull_plan`, no network)

`<model dir>/.oflm-files.json` maps each file to the registry oid it was downloaded for. A pull
adds each file as its download completes, so an interrupted pull keeps what it finished. The
record holds the registry's oid, not a hash of the bytes. `oflm remove` deletes it with the model.

A file that is present, the right size and not named in the record (every file of an install made
before the record existed) is hashed once by the next pull. A match is recorded without a download;
a mismatch is downloaded. That catches a file that changed but kept its size, which no status check
can see. Until then, `oflm list` and the automatic pull treat such an install by size alone; an
explicit `oflm pull <tag>` runs the one-time check even when the status is Ready.

The one-time check reads every file: klein's 9 GB took 96 s (2026-10-08, `picosha2`, about
95 MB/s). After that, a pull of a current model returns at once.

**Acceptance criteria:**
- Of four files (one matching, one the same size with other bytes, one the wrong size, one
  absent), a pull with no record fetches the last three and records the first.
- A recorded file is not hashed again: its record, not its bytes, says it is current.
- A file the registry gives no oid is neither hashed nor fetched for that reason.
- The record reads back what was written; an absent or unreadable record reads as no record.

### PULL-FORCE: `--force` downloads every file
**Applies to:** openflowlm-next (`src/pull/install_record.cpp`, `src/pull/model_downloader.cpp`)
**Verification:** test
**Test:** `src/pull/pull_plan_test.cpp` (CTest `pull_plan`, no network)

`oflm pull <tag> --force` fetches every file the model lists, present and current ones included,
whatever the status.

**Acceptance criteria:**
- With `force`, every file is planned, each for `--force`.

### PULL-ATOMIC: a download replaces a file only once it is whole
**Applies to:** openflowlm-next (`src/pull/download_model.cpp`, `src/pull/model_downloader.cpp`)
**Verification:** manual

A file downloads to `<file>.part` and is renamed over the old file only when the transfer
succeeds; a failure removes the `.part`. A pull killed mid-file leaves the old file in place and
its `.part` behind, and the next pull removes the `.part` before anything else. A replacement needs
disk for the largest file twice, briefly (7.5 GB for klein's `weights.bin`).

**Verification (manual):** on a complete install, `oflm pull <tag> --force` and kill it while it
downloads the largest file. The old file is still there with the registry's size and hash, and a
`.part` is beside it. The next `oflm pull <tag>` removes the `.part` and reports the model already
downloaded.

(2026-10-08, klein: killed with 1.13 GB of `weights.bin.part` written. `weights.bin` kept
the registry's size and sha256, and the next pull removed the `.part` and returned at once.)

### PULL-UPGRADE: an installed model upgrades with one pull
**Applies to:** openflowlm-next (`src/pull/`)
**Verification:** manual

An install of an earlier registry revision becomes the new revision with one `oflm pull <tag>`,
with no `oflm remove` and no `--force`. Exactly the files that differ are downloaded.

**Verification (manual):** copy the 17 files of #136's `flux2-klein:4b` (registry revision `main`
before #137) into an empty model root. With `OFLM_MODEL_PATH` at that root and `OFLM_CONFIG_PATH`
at #137's `model_list.json` (revision `d84e3fd6`), run `oflm pull flux2-klein:4b`. It downloads
the 4 files whose size changed, the one that changed with the same size (`vae_W.bin`) and the 4 new
ones, and keeps the other 12. #137's `oflm image` then writes the same bytes from this install as
from a fresh pull.

(2026-10-08: the pull took 93 s, most of it the one-time hash, and downloaded those 9 files,
128 MB. `vae_W.bin` was caught by the hash. The 512² image for seed 1 was the same bytes as a
fresh install's. A second pull returned at once.)

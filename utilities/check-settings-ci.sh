#!/usr/bin/env bash
# Run the settings-registry CI check the way CI runs it.
#
# CI merges the pull request with its target branch, regenerates the settings
# registry, and fails if the generated files differ from the committed ones.
# The interesting part is the merge: a branch that adds an OFLM_* variable, and
# a target branch that adds one while your branch is open, both break the
# check -- and the second one has nothing to do with your change.
#
# This script does the merge in a scratch worktree, so your working tree is
# left alone.
#
# Usage:  utilities/check-settings-ci.sh [target-branch]
#         utilities/check-settings-ci.sh origin/main
#
# The target branch defaults to origin/main. It is fetched over the remote's
# URL, so it works whether the remote is SSH or HTTPS.

set -uo pipefail
exec 2>&1

TARGET="${1:-origin/main}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Only a MODIFIED tracked file blocks: an untracked one is not part of the
# branch, and CI never sees it.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "check-settings-ci: the working tree has changes; commit or stash them first."
  git status --short --untracked-files=no
  exit 2
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "check-settings-ci: branch $BRANCH, target $TARGET"

# ---- fetch the target branch over the remote's own URL --------------------
# SSH is not always usable (no key, no agent); the URL works with whatever
# credentials git already has.
remote="$(git config --get "branch.$BRANCH.remote" || echo origin)"
url="$(git config --get "remote.$remote.url" || echo https://github.com/Atomic-Germ/OpenFlowLM-Next.git)"
name="$(basename "$TARGET")"
if timeout 300 git fetch --quiet "$url" "+refs/heads/$name:refs/remotes/ci-check/$name" 2>/dev/null; then
  TARGET="refs/remotes/ci-check/$name"
  echo "check-settings-ci: fetched $name"
else
  echo "check-settings-ci: could not fetch $name; using the local copy of $TARGET"
fi

BASE="${TARGET#refs/remotes/}"
if ! git rev-parse --verify "$BASE" >/dev/null 2>&1; then
  echo "check-settings-ci: $TARGET is not known locally; fetch it or pass another branch."
  exit 2
fi

# ---- the merge, in a scratch worktree -------------------------------------
WORK="$(mktemp -d)"
SCRATCH="$WORK/merge"
cleanup() {
  git worktree remove --force "$SCRATCH" >/dev/null 2>&1
  rm -rf "$WORK"
}
trap cleanup EXIT

git worktree add --detach "$SCRATCH" HEAD >/dev/null 2>&1 || {
  echo "check-settings-ci: could not create the scratch worktree"
  exit 2
}

echo "check-settings-ci: merging $BASE into the branch"
if ! git -C "$SCRATCH" merge --no-edit "$BASE" >/dev/null 2>&1; then
  echo
  echo "check-settings-ci: the merge CONFLICTS."
  echo "  Merge $BASE into your branch and settle it -- CI merges it too, and a"
  echo "  conflict there is a conflict in the PR."
  echo "  The files that clash are:"
  git -C "$SCRATCH" merge --no-edit "$BASE" 2>&1 | grep -E "CONFLICT|conflict" | head -10
  exit 1
fi

# ---- the check CI runs ----------------------------------------------------
echo "check-settings-ci: regenerating the settings registry"
if ! cmake -S src -B "$SCRATCH/build" >/dev/null 2>&1; then
  echo "check-settings-ci: could not configure the scratch build"
  exit 2
fi
if ! cmake --build "$SCRATCH/build" --target generate-settings 2>&1 | grep -vE "^\[|^ninja|Generating|Generated" | head -20; then
  :
fi

# The generator writes into the source tree it was given.
if ! cmake --build "$SCRATCH/build" --target generate-settings >/dev/null 2>&1; then
  echo
  echo "check-settings-ci: the generator FAILED on the merged tree."
  echo "  Run it yourself to see what it says:"
  echo "    cmake --build --preset linux-default --target generate-settings"
  echo "  Every CLI flag and every OFLM_* variable the code reads must be a"
  echo "  setting in src/config/settings.json, or be listed in"
  echo "  NON_SETTING_ENV_VARS / NON_SETTING_FLAGS in utilities/generate-registry.py"
  echo "  with a reason."
  exit 1
fi

# The generated files must match the committed ones.
if ! git -C "$SCRATCH" diff --exit-code -- src/config/settings.json \
                                      src/config/settings_registry_generated.hpp >/dev/null; then
  echo
  echo "check-settings-ci: the generated files differ from the committed ones."
  echo "  The merged tree produces a different registry -- copy the result over:"
  echo "    cp $SCRATCH/src/config/settings.json \\"
  echo "       $SCRATCH/src/config/settings_registry_generated.hpp src/config/"
  echo "  and commit it."
  git -C "$SCRATCH" diff --stat -- src/config/settings.json \
                                    src/config/settings_registry_generated.hpp
  exit 1
fi

echo
echo "check-settings-ci: OK -- the merged tree regenerates the committed registry."

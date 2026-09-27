#!/usr/bin/env bash
# Check a packaged OpenFlowLM distribution: the DEB, the RPM and the TGZ.
#
#   utilities/release/verify-package.sh build/packages [--version X.Y.Z]
#                                                    [--require deb,rpm,tgz]
#                                                    [--expect-kernels glob]
#
# WHY A SCRIPT AND NOT A LOOK. A package that builds is not a package that
# works, and the ways this one breaks quietly are specific:
#
#   * The xclbin tree is installed by `install(DIRECTORY xclbins ...)`, which
#     installs whatever is in the source tree at install time. A build without
#     the prebuilt kernel sets therefore produces a package that builds, packs,
#     installs, starts, and then cannot load a single open model -- because the
#     only xclbins in it are the closed ones git already tracks.
#   * The engine's shared libraries are globbed with `file(GLOB ...)` plus a
#     filter for `.bak`, and a backup of libq4_npu_eXpress.so is tracked in the
#     repo. A glob change that drops the filter ships a second, older engine
#     library beside the new one, and which one loads depends on the loader.
#   * oflm has to be ON PATH after install (the package writes /usr/bin/oflm and
#     /etc/profile.d/openflowlm.sh) or the first command in every support doc
#     fails.
#   * The DEB's Depends and the RPM's Requires have to name the NPU runtime. XRT
#     is a system dependency, not something the package carries: a package with
#     no dependency on it installs happily on a machine with no driver and then
#     fails to open a device.
#
# Assert all of it, per format, and fail the release if any of it is absent.
# Every list is normalised to the same shape first (no leading "./", no tar
# top-level directory), so one set of path needles checks all three formats.

set -euo pipefail

DIR="${1:-build/packages}"
[ $# -gt 0 ] && shift
VERSION=""
# The two prebuilt families, as regexes over the package's own path list. Both
# are required: utilities/export-kernels.py builds them in one run, so one
# without the other means an incomplete bundle, not a deliberate omission.
# Overridable because the naming is a .gitignore convention and a new family
# should not need an edit here to be checked at all.
EXPECT_KERNELS='xclbins/[^/]+/open_kernels[^/]*/.*\.xclbin$'
EXPECT_BERT='xclbins/BERT-h[^/]*/.*\.xclbin$'
# Formats the caller insists on, e.g. "deb,rpm,tgz" from the release workflow.
# Without it, a directory holding only a TGZ is checked as "whatever is here",
# which is what a local single-format build wants -- and also what a cpack that
# silently produced one format instead of three would look like.
REQUIRE=""
FAILURES=0

while [ $# -gt 0 ]; do
    case "$1" in
        --version)        VERSION="${2:?}"; shift 2 ;;
        --require)        REQUIRE="${2:?}"; shift 2 ;;
        --expect-kernels) EXPECT_KERNELS="${2:?}"; shift 2 ;;
        --expect-bert)    EXPECT_BERT="${2:?}"; shift 2 ;;
        -h|--help)        sed -n '3,24p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[ -d "$DIR" ] || { echo "ERROR: no such directory: $DIR" >&2; exit 1; }

fail() { echo "  FAIL: $*"; FAILURES=$((FAILURES + 1)); }
pass() { echo "  ok:   $*"; }

# A required format that produced no file at all. The loops below would glob
# literally and then hand the literal to dpkg-deb/rpm, which fails with a message
# about a file called "*.deb" -- technically an error, practically a puzzle.
check_required() {
    local want
    IFS=',' read -ra want <<<"$REQUIRE"
    local f ext
    for f in "${want[@]}"; do
        # cpack writes .tar.gz, not .tgz. Accept both spellings of the same
        # format rather than failing a release over a suffix.
        case "$f" in tgz) ext="tar.gz" ;; *) ext="$f" ;; esac
        # shellcheck disable=SC2053  # a glob is the point
        if ! compgen -G "$DIR/*.$ext" >/dev/null; then
            fail "no .$ext in $DIR, and --require asked for one. (cpack -G only honours" \
                 "the LAST -G on one command line; run it once per format.)"
        fi
    done
}

# The shared expectations. Directories carry a trailing slash in some formats
# and not in others, so they are matched with has_dir.
check_common() {
    local label="$1" files="$2"
    has_path "$label" "$files" "usr/bin/oflm"                       "oflm is on PATH"
    has_path "$label" "$files" "etc/profile.d/openflowlm.sh"        "login shells get the prefix"
    has_path "$label" "$files" "opt/openflowlm/bin/oflm"            "the engine"
    has_path "$label" "$files" "opt/openflowlm/lib64/libq4_npu_eXpress.so" \
                                                                     "the engine libraries"
    has_path "$label" "$files" "opt/openflowlm/share/oflm/model_list.json" \
                                                                     "the model registry"
    has_dir  "$label" "$files" "opt/openflowlm/share/oflm/xclbins"  "the kernel sets"
    has_path "$label" "$files" "opt/openflowlm/share/oflm/oflm-add/oflm-add.py" \
                                                                     "oflm-add"
    has_dir  "$label" "$files" "opt/openflowlm/share/oflm/utilities/oflm-test" \
                                                                     "oflm-test"
    has_kernels "$label" "$files"
    no_backups "$label" "$files"
}

has_path() {  # label files needle description
    if grep -qxF -- "$3" <<<"$2"; then pass "$4"; else fail "$4 (missing: $3)"; fi
}

has_dir() {   # label files needle description
    if grep -qxF -- "$3" <<<"$2" || grep -qxF -- "$3/" <<<"$2"; then
        pass "$4"
    else
        fail "$4 (missing: $3/)"
    fi
}

has_kernels() {
    # Both families, and matched on the FILE not the path: share/oflm/
    # open_kernels/recipes/*.py is the q4nx recipe module and is tracked in
    # git, so a check that greps for "open_kernels" anywhere passes on a build
    # with no prebuilt kernel in it at all.
    local open bert
    open="$(grep -cE "$EXPECT_KERNELS" <<<"$2" || true)"
    bert="$(grep -cE "$EXPECT_BERT" <<<"$2" || true)"
    if [ "$open" -gt 0 ]; then
        pass "ships $open open-kernel xclbin file(s) ($EXPECT_KERNELS)"
    else
        fail "ships NO open kernel xclbins. Nothing git ignores was in src/xclbins," \
             "so only the closed sets are in here and no open model will load." \
             "Stage the prebuilts (utilities/release/stage-prebuilts.sh) and re-run."
    fi
    if [ "$bert" -gt 0 ]; then
        pass "ships $bert BERT design-set xclbin(s) ($EXPECT_BERT)"
    else
        fail "ships no BERT design sets. Those are the open_npue embedding kernels," \
             "and utilities/export-kernels.py builds them in the same run as the" \
             "open kernel sets, so their absence means the prebuilt bundle is" \
             "incomplete rather than that this release skips them."
    fi
}

no_backups() {
    if grep -q -- '\.bak' <<<"$2"; then
        fail "ships a .bak file: $(grep -- '\.bak' <<<"$2" | head -3 | tr '\n' ' ')" \
             "(the engine-lib glob in src/CMakeLists.txt lost its .bak filter)"
    else
        pass "no .bak files"
    fi
}

check_required

# ------------------------------------------------------------------ the DEB
shopt -s nullglob
for deb in "$DIR"/*.deb; do
    echo "== $(basename "$deb")"
    # `dpkg-deb -c` prints "path -> target" for a symlink, so cut the arrow
    # before taking the last field: ./usr/bin/oflm -> /opt/... would otherwise
    # look like a file at /opt/ and miss the PATH entry entirely.
    files="$(dpkg-deb --contents "$deb" | sed 's/ *-> .*$//' | awk '{print $NF}' | sed 's|^\./||')"
    check_common "$(basename "$deb")" "$files"
    if dpkg-deb --field "$deb" Depends | grep -q 'libxrt-npu2'; then
        pass "Depends names the NPU runtime (libxrt-npu2)"
    else
        fail "Depends does not name libxrt-npu2; this DEB installs on a machine" \
             "with no NPU driver and then fails to open a device"
    fi
    if [ -n "$VERSION" ]; then
        got="$(dpkg-deb --field "$deb" Version)"
        # Not `a && pass || fail`: pass is a printf, and the day it is not, the
        # fail runs too and the release fails for the right reason by accident.
        if [ "$got" = "$VERSION" ]; then
            pass "Version is $VERSION"
        else
            fail "Version is $got, not $VERSION"
        fi
    fi
done

# ------------------------------------------------------------------ the RPM
for rpm in "$DIR"/*.rpm; do
    echo "== $(basename "$rpm")"
    # rpm prints absolute paths ("/opt/..."), dpkg-deb and tar print relative
    # ones, and the needles below are relative and matched as whole lines. Left
    # unnormalised, every check in check_common fails on the RPM alone -- which
    # reads like a broken package rather than a prefix in a filename.
    files="$(rpm -qlp "$rpm" | sed 's|^/||')"
    check_common "$(basename "$rpm")" "$files"
    if rpm -qp --requires "$rpm" 2>/dev/null | grep -q 'libxrt_coreutil'; then
        pass "Requires the NPU runtime (libxrt_coreutil, via AUTOREQ)"
    else
        fail "does NOT require libxrt_coreutil. CPACK_RPM_PACKAGE_AUTOREQ is off," \
             "or the engine stopped linking XRT, and this RPM installs on a" \
             "machine with no driver."
    fi
    if [ -n "$VERSION" ]; then
        got="$(rpm -qp --qf '%{VERSION}' "$rpm")"
        if [ "$got" = "$VERSION" ]; then
            pass "Version is $VERSION"
        else
            fail "Version is $got, not $VERSION"
        fi
    fi
done

# ------------------------------------------------------------------ the TGZ
for tgz in "$DIR"/*.tar.gz; do
    case "$tgz" in *-nix.tar.gz) continue ;; esac   # the flake, not the payload
    echo "== $(basename "$tgz")"
    # Strip CPack's top-level directory (openflowlm-<version>-Linux) so these
    # paths are comparable with the other two formats'.
    files="$(tar tzf "$tgz" | sed 's|^[^/]*/||')"
    check_common "$(basename "$tgz")" "$files"
done

echo
if [ "$FAILURES" -gt 0 ]; then
    echo "verify-package.sh: $FAILURES check(s) failed" >&2
    exit 1
fi
echo "verify-package.sh: all checks passed"

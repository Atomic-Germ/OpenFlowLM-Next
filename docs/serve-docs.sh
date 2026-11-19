#!/usr/bin/env bash
#
# serve-docs.sh - build or preview the OpenFlowLM Hugo Lotus Docs site locally.
#
# Uses Hugo Extended (v0.147.9). If a compatible Hugo Extended (>= 0.140.0) is
# not on PATH, automatically downloads Hugo Extended v0.147.9 for Linux or
# macOS into ${XDG_CACHE_HOME:-$HOME/.cache}/oflm-hugo/v0.147.9/hugo.
#
# Usage:
#   ./docs/serve-docs.sh           # Serve with live reload at http://127.0.0.1:1313
#   PORT=8080 ./docs/serve-docs.sh # Serve on a custom port
#   ./docs/serve-docs.sh --build   # Build minified static site into docs/public/
#
set -euo pipefail

HUGO_VERSION="0.147.9"
DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${DOCS_DIR}/hugo.toml" ]; then
  echo "ERROR: hugo.toml not found in ${DOCS_DIR}" >&2
  exit 1
fi

is_valid_hugo() {
  local bin="$1"
  if [ ! -x "${bin}" ]; then
    return 1
  fi
  local ver_out
  ver_out="$("${bin}" version 2>/dev/null || true)"
  if [[ "${ver_out}" != *"+extended"* ]] && [[ "${ver_out}" != *"extended"* ]]; then
    return 1
  fi
  if [[ "${ver_out}" =~ v([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
    local major="${BASH_REMATCH[1]}"
    local minor="${BASH_REMATCH[2]}"
    if (( major > 0 || minor >= 140 )); then
      return 0
    fi
  fi
  return 1
}

resolve_hugo() {
  if command -v hugo >/dev/null 2>&1; then
    local sys_hugo
    sys_hugo="$(command -v hugo)"
    if is_valid_hugo "${sys_hugo}"; then
      printf '%s\n' "${sys_hugo}"
      return 0
    fi
  fi

  local cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/oflm-hugo/v${HUGO_VERSION}"
  local cached_bin="${cache_root}/hugo"
  if is_valid_hugo "${cached_bin}"; then
    printf '%s\n' "${cached_bin}"
    return 0
  fi

  local os arch platform_slug
  os="$(uname -s)"
  arch="$(uname -m)"
  case "${os}" in
    Linux)
      case "${arch}" in
        x86_64|amd64) platform_slug="linux-amd64" ;;
        aarch64|arm64) platform_slug="linux-arm64" ;;
        *)
          echo "ERROR: Unsupported Linux architecture '${arch}' for automatic Hugo download." >&2
          exit 1
          ;;
      esac
      ;;
    Darwin)
      platform_slug="darwin-universal"
      ;;
    *)
      echo "ERROR: Unsupported OS '${os}'. Install Hugo Extended >= 0.140.0 manually." >&2
      exit 1
      ;;
  esac

  local archive_url="https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_${platform_slug}.tar.gz"
  echo "Downloading Hugo Extended v${HUGO_VERSION} (${platform_slug}) to ${cached_bin} ..." >&2
  mkdir -p "${cache_root}"
  local tmp_tar="${cache_root}/hugo.tar.gz"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${archive_url}" -o "${tmp_tar}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${tmp_tar}" "${archive_url}"
  else
    echo "ERROR: Neither curl nor wget is available to download Hugo." >&2
    exit 1
  fi

  tar -xzf "${tmp_tar}" -C "${cache_root}" hugo
  rm -f "${tmp_tar}"
  chmod +x "${cached_bin}"

  if ! is_valid_hugo "${cached_bin}"; then
    echo "ERROR: Downloaded Hugo binary at ${cached_bin} failed verification." >&2
    exit 1
  fi

  printf '%s\n' "${cached_bin}"
}

if ! command -v go >/dev/null 2>&1; then
  echo "WARNING: 'go' was not found on PATH. Hugo Go modules (Lotus Docs) require Go >= 1.21 if modules are not cached." >&2
fi

HUGO_BIN="$(resolve_hugo)"
echo "Using $("${HUGO_BIN}" version)"

if [ "${1:-}" = "--build" ]; then
  shift
  echo "Building minified site into ${DOCS_DIR}/public ..."
  exec "${HUGO_BIN}" --source "${DOCS_DIR}" --minify "$@"
else
  PORT="${PORT:-1313}"
  echo "Serving OpenFlowLM docs at http://127.0.0.1:${PORT} ..."
  exec "${HUGO_BIN}" server --source "${DOCS_DIR}" --bind 127.0.0.1 --port "${PORT}" --buildFuture "$@"
fi
{
  srcRoot,
  lib,
  stdenv,
  fetchurl,
  python312,
  python312Packages,
  pkgs,
  xrt,
  git,
  makeWrapper,
  unzip,
  zlib,
  patchelf,
  # BERT embedding sets need pyxrt + an NPU at build time.  In a sandboxed
  # Nix build that means __noChroot, which is not allowed when sandbox=true.
  # Default to skipping BERT so the package builds everywhere; NPU hosts can
  # override with skipBert = false (or use the openflowlm-open-kernels-with-bert
  # / oflm-with-bert flake packages).
  skipBert ? true,
  # No spec is skipped by default. An earlier revision of this derivation
  # skipped qwen25-3b, minicpm5-2b and phi4-mini-4b, which failed with the
  # mlir-aie 1.4.3 / Peano 20260923 toolchain pinned in open-kernels-env.nix
  # (qwen25-3b: no bias stream in dx_attn; the other two: an aiecc
  # "aie.objectfifo.pool op segment 0 has no filler" placement error). Both are
  # recipe problems that have since been fixed upstream, and all 13 specs build
  # today -- re-add a name here if a future recipe regresses.
  skipSpecs ? "",
}:

let
  env = lib.callPackageWith (
    {
      inherit lib stdenv fetchurl python312 python312Packages xrt git makeWrapper unzip zlib patchelf pkgs;
    }
  ) ./open-kernels-env.nix {};
in

stdenv.mkDerivation rec {
  pname = "openflowlm-open-kernels";
  version = "0.1.0";

  src = srcRoot;

  nativeBuildInputs = env.nativeTools ++ [ env.ironvenv ];
  buildInputs = env.runtimeLibs;

  buildPhase = ''
    export HOME=$TMPDIR
    export XILINX_XRT="${env.xrtCombined}"
    export OFLM_VENV_DIR="$TMPDIR/ironvenv"
    export OFLM_SKIP_VENV_SETUP=1

    # The export writes build directories (open_kernels/designs/*/build) and the
    # finished sets (src/xclbins) into the source tree, which a Nix store path
    # does not allow. Build from a writable copy under $TMPDIR instead of
    # pointing the scripts at a second, hand-maintained checkout.
    buildTree="$TMPDIR/openflowlm"
    export buildTree
    cp -R . "$buildTree"
    chmod -R u+w "$buildTree"
    # src/xclbins is git-ignored, so a developer checkout carries whatever they
    # last built locally; copying it would ship those sets into the store next
    # to this build's output. Start from an empty tree.
    rm -rf "$buildTree/src/xclbins"
    cd "$buildTree"

    # utilities/export-kernels.py expects a writable venv.  Materialize it
    # under $TMPDIR so the source tree is never modified (important both for
    # sandbox builds and for local dev shells that run from a writable git
    # checkout), and tell the script where to find it via OFLM_VENV_DIR.
    cp -R ${env.ironvenv} "$OFLM_VENV_DIR"
    chmod -R +w "$OFLM_VENV_DIR"
    ln -sf ${python312}/bin/python "$OFLM_VENV_DIR/bin/python"

    export PEANO_INSTALL_DIR="$OFLM_VENV_DIR/${env.env.PEANO_INSTALL_DIR}"
    export PATH="$OFLM_VENV_DIR/${env.pySite}/llvm-aie/bin:$OFLM_VENV_DIR/${env.pySite}/mlir_aie/bin:${env.xrtCombined}/bin:$PATH"
    export PYTHONPATH="$OFLM_VENV_DIR/${env.pySite}:${env.xrtCombined}/python''${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="${env.env.LD_LIBRARY_PATH}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # The manylinux llvm-aie/mlir-aie wheels ship x86_64 executables that
    # hard-code /lib64/ld-linux-x86-64.so.2.  In the Nix sandbox /lib64 is
    # unavailable, so patch the interpreter to the stdenv linker and add the
    # C++ runtime + zlib to the RUNPATH without destroying the wheel's own
    # library search paths (e.g. mlir_aie.libs/libcrypto-...so.3).
    for dir in "$PEANO_INSTALL_DIR/bin" "$OFLM_VENV_DIR/${env.pySite}/mlir_aie/bin"; do
      if [ -d "$dir" ]; then
        for bin in "$dir/"*; do
          if [ -f "$bin" ] && [ -x "$bin" ]; then
            ${patchelf}/bin/patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" "$bin" 2> /dev/null || true
            ${patchelf}/bin/patchelf --add-rpath '${stdenv.cc.cc.lib}/lib:${zlib}/lib' "$bin" 2> /dev/null || true
          fi
        done
      fi
    done

    # Skip whatever skipSpecs names (see the derivation's argument for why).
    python utilities/export-kernels.py --force --jobs "''${NIX_BUILD_CORES:-4}" \
      ${lib.optionalString skipBert "--skip-bert"} \
      ${lib.optionalString (skipSpecs != "") "--skip-specs ${skipSpecs}"}
  '';

  installPhase = ''
    mkdir -p $out/share/oflm
    cp -r "$buildTree/src/xclbins" $out/share/oflm/
  '';

  # The open_npue BERT embedding sets need pyxrt and an NPU at build time.
  # Build with skipBert=false only on a machine that exposes /dev/accel* to
  # the Nix builder and has sandbox disabled or relaxed for this derivation.
  __noChroot = !skipBert;

  meta = {
    description = "OpenFlowLM open NPU kernel xclbins";
    platforms = [ "x86_64-linux" ];
  };
}

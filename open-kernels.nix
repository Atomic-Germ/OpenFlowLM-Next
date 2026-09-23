{
  lib,
  stdenv,
  fetchurl,
  python312,
  python312Packages,
  xrt,
  git,
  makeWrapper,
  unzip,
  zlib,
  patchelf,
  skipBert ? true,
  skipSpecs ? "qwen25-3b,minicpm5-2b,phi4-mini-4b",
}:

let
  env = lib.callPackageWith (
    {
      inherit lib stdenv fetchurl python312 python312Packages xrt git makeWrapper unzip zlib patchelf;
    }
  ) ./open-kernels-env.nix {};
in

stdenv.mkDerivation rec {
  pname = "openflowlm-open-kernels";
  version = "0.1.0";

  src = lib.cleanSourceWith {
    src = ./.;
    filter = path: type:
      let
        base = baseNameOf path;
        isBuildOutput = lib.hasSuffix ".o" base
          || lib.hasSuffix ".a" base
          || lib.hasSuffix ".so" base
          || lib.hasSuffix ".dylib" base
          || lib.hasSuffix ".dll" base
          || base == "result" || base == "result-bin";
        isEngineLib = lib.hasInfix "/src/lib/" path && (lib.hasSuffix ".so" base || lib.hasSuffix ".so.bak" base);
      in
        (type == "directory" || !isBuildOutput || isEngineLib)
        && !(base == ".git");
  };

  nativeBuildInputs = env.nativeTools ++ [ env.ironvenv ];
  buildInputs = env.runtimeLibs;

  buildPhase = ''
    export HOME=$TMPDIR
    export XILINX_XRT="${xrt}/opt/xilinx/xrt"

    # utilities/export-kernels.py expects a writable venv at ./ironvenv with
    # its own bin/python. Copy the nix-provided venv and add a python symlink.
    rm -rf ironvenv
    cp -R ${env.ironvenv} ironvenv
    chmod -R +w ironvenv
    ln -sf ${python312}/bin/python ironvenv/bin/python

    export PEANO_INSTALL_DIR="$PWD/ironvenv/${env.env.PEANO_INSTALL_DIR}"
    export PATH="$PWD/ironvenv/${env.pySite}/llvm-aie/bin:$PWD/ironvenv/${env.pySite}/mlir_aie/bin:${xrt}/opt/xilinx/xrt/bin:$PATH"
    export PYTHONPATH="$PWD/ironvenv/${env.pySite}:${xrt}/opt/xilinx/xrt/python''${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="${env.env.LD_LIBRARY_PATH}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # The manylinux llvm-aie/mlir-aie wheels ship x86_64 executables that
    # hard-code /lib64/ld-linux-x86-64.so.2.  In the Nix sandbox /lib64 is
    # unavailable, so patch the interpreter to the stdenv linker and add the
    # C++ runtime + zlib to the RUNPATH without destroying the wheel's own
    # library search paths (e.g. mlir_aie.libs/libcrypto-...so.3).
    for dir in "$PEANO_INSTALL_DIR/bin" "$PWD/ironvenv/${env.pySite}/mlir_aie/bin"; do
      if [ -d "$dir" ]; then
        for bin in "$dir/"*; do
          if [ -f "$bin" ] && [ -x "$bin" ]; then
            ${patchelf}/bin/patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" "$bin" 2> /dev/null || true
            ${patchelf}/bin/patchelf --add-rpath '${stdenv.cc.cc.lib}/lib:${zlib}/lib' "$bin" 2> /dev/null || true
          fi
        done
      fi
    done

    # A few specs fail on dx_attn with the current recipe/toolchain:
    #   - qwen25-3b: attention bias not supported by this design
    #   - minicpm5-2b, phi4-mini-4b: aiecc objectfifo.pool placement error
    # Skip them until the open_kernels recipe covers them.
    python utilities/export-kernels.py --force --jobs "''${NIX_BUILD_CORES:-4}" \
      ${lib.optionalString skipBert "--skip-bert"} \
      ${lib.optionalString (skipSpecs != "") "--skip-specs ${skipSpecs}"}
  '';

  installPhase = ''
    mkdir -p $out/share/oflm
    cp -r src/xclbins $out/share/oflm/
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

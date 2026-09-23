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
  pyVersion = "3.12";
  pySite = "lib/python${pyVersion}/site-packages";

  mlirAieWheel = fetchurl {
    url = "https://github.com/Xilinx/mlir-aie/releases/download/v1.4.3/mlir_aie-1.4.3-cp312-cp312-manylinux_2_35_x86_64.whl";
    sha256 = "sha256-TYwjpbYXHdEWLHjFYPCkjK1llpC1vlMNUWZrf/9ScoE=";
  };

  llvmAieWheel = fetchurl {
    url = "https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026092301%2B02be6fd8-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    sha256 = "sha256-bi0uvoa/Z12aGcIhzUMCMtMUw5EEiUTeP8QVB5JANK4=";
  };

  pyDeps = with python312Packages; [
    aiofiles
    cloudpickle
    mdurl
    ml-dtypes
    numpy
    pygments
    rich
  ];

  ironvenv = stdenv.mkDerivation {
    pname = "openflowlm-ironvenv";
    version = "1.4.3-20260923";

    nativeBuildInputs = [ python312 unzip ];
    src = ./.;
    dontUnpack = true;

    installPhase = ''
      ${python312}/bin/python -m venv $out
      export SITE="$out/${pySite}"

      ${unzip}/bin/unzip -q -d "$SITE" ${mlirAieWheel}
      ${unzip}/bin/unzip -q -d "$SITE" ${llvmAieWheel}

      # Link nixpkgs Python deps into the venv so imports resolve without
      # re-installing them or relying on .pth files.
      ${lib.concatStringsSep "\n" (map (pkg: "ln -s ${pkg}/${pySite}/* \"$SITE/\"") pyDeps)}

      # The wheel drops the aie package under mlir_aie/python; make it
      # importable without relying on aie.pth.
      ln -s "$SITE/mlir_aie/python/aie" "$SITE/aie"

      mkdir -p $out/nix-support
      echo "OpenFlowLM IRON toolchain venv (Python 3.12)" > $out/nix-support/hydra-build-products
    '';

    meta.platforms = [ "x86_64-linux" ];
  };
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

  nativeBuildInputs = [ git makeWrapper python312 stdenv.cc.cc.lib patchelf ];
  buildInputs = [ xrt ironvenv ];

  buildPhase = ''
    export HOME=$TMPDIR
    export XILINX_XRT="${xrt}/opt/xilinx/xrt"

    # utilities/export-kernels.py expects a writable venv at ./ironvenv with
    # its own bin/python. Copy the nix-provided venv and add a python symlink.
    rm -rf ironvenv
    cp -R ${ironvenv} ironvenv
    chmod -R +w ironvenv
    ln -sf ${python312}/bin/python ironvenv/bin/python

    export PEANO_INSTALL_DIR="$PWD/ironvenv/${pySite}/llvm-aie"
    export PATH="$PEANO_INSTALL_DIR/bin:${xrt}/opt/xilinx/xrt/bin:$PATH"
    export PYTHONPATH="$PWD/ironvenv/${pySite}:${xrt}/opt/xilinx/xrt/python''${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="${lib.makeLibraryPath [ stdenv.cc.cc.lib zlib xrt ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # The manylinux llvm-aie/mlir-aie wheels ship x86_64 executables that
    # hard-code /lib64/ld-linux-x86-64.so.2.  In the Nix sandbox /lib64 is
    # unavailable, so patch the interpreter to the stdenv linker and add the
    # C++ runtime + zlib to the RUNPATH without destroying the wheel's own
    # library search paths (e.g. mlir_aie.libs/libcrypto-...so.3).
    for dir in "$PEANO_INSTALL_DIR/bin" "$PWD/ironvenv/${pySite}/mlir_aie/bin"; do
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

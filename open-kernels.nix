{
  lib,
  stdenv,
  fetchurl,
  python311,
  xrt,
  git,
  makeWrapper,
  unzip,
  zlib,
  patchelf,
  skipBert ? true,
  skipSpecs ? "qwen25-3b",
}:

let
  mlirAieWheel = fetchurl {
    url = "https://github.com/Xilinx/mlir-aie/releases/download/v1.4.2/mlir_aie-1.4.2-cp311-cp311-manylinux_2_35_x86_64.whl";
    sha256 = "1kfr46p6bl70zc9bw6s68z4prn6r4hli8srr1vzssq824lh4x10p";
  };

  llvmAieWheel = fetchurl {
    url = "https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-21.0.0.2026080301%2Bc9c5ecb7-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    sha256 = "0x8kkgl6cvr389as6aabw6vbvwknm4ghqkn39l508i3ja72jfv2s";
  };

  pyWheels = [
    (fetchurl {
      url = "https://files.pythonhosted.org/packages/py3/a/aiofiles/aiofiles-24.1.0-py3-none-any.whl";
      sha256 = "1rb0haxzh3lsafw1y8sl97fn9s332w37xgyimgbvagjy37s5bv5l";
    })
    (fetchurl {
      url = "https://files.pythonhosted.org/packages/py3/c/cloudpickle/cloudpickle-3.1.2-py3-none-any.whl";
      sha256 = "0jmz3yz0dcjws9kl3j565zs5zw3jnh0vhfzr3pf60gypmzv4gjws";
    })
    (fetchurl {
      url = "https://files.pythonhosted.org/packages/py3/m/mdurl/mdurl-0.1.2-py3-none-any.whl";
      sha256 = "1y5qjqhmq2nm7xj6w5rrp503r7jhj7zr2qcnr6gs858nwm0ql044";
    })
    (fetchurl {
      url = "https://files.pythonhosted.org/packages/cp311/m/ml-dtypes/ml_dtypes-0.5.4-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      sha256 = "10ql7w2s02cki6mxgc7fw7way3y2hfkqmnpvl8z4a7pjk0ssbf8r";
    })
    (fetchurl {
      url = "https://files.pythonhosted.org/packages/cp311/n/numpy/numpy-2.2.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
      sha256 = "1pxvdmz8klm0wv6cvimh2lfa0g3xlwaf0cqqaa543z4q310zh45s";
    })
    (fetchurl {
      url = "https://files.pythonhosted.org/packages/py3/p/pygments/pygments-2.19.1-py3-none-any.whl";
      sha256 = "133dmda902c3wcg63c1lf1jwxfwkbb9nvarg4jwg9v2wsm5598cy";
    })
    (fetchurl {
      url = "https://files.pythonhosted.org/packages/py3/r/rich/rich-14.0.0-py3-none-any.whl";
      sha256 = "1q6pjp1qs1l3dqzrj57y7y95hknhwf748bylzz50kb0sjphr350w";
    })
  ];

  ironvenv = stdenv.mkDerivation {
    pname = "openflowlm-ironvenv";
    version = "1.4.2-20260803";

    nativeBuildInputs = [ python311 unzip ];
    src = ./.;
    dontUnpack = true;

    installPhase = ''
      ${python311}/bin/python -m venv $out
      export SITE="$out/lib/python3.11/site-packages"

      ${unzip}/bin/unzip -q -d "$SITE" ${mlirAieWheel}
      ${unzip}/bin/unzip -q -d "$SITE" ${llvmAieWheel}
      ${lib.concatStringsSep "\n" (map (wheel: "${unzip}/bin/unzip -q -d \"$SITE\" ${wheel}") pyWheels)}

      # The wheel drops the aie package under mlir_aie/python; make it
      # importable without relying on aie.pth, which nix venv symlinking can
      # break.
      ln -s "$SITE/mlir_aie/python/aie" "$SITE/aie"

      mkdir -p $out/nix-support
      echo "OpenFlowLM IRON toolchain venv" > $out/nix-support/hydra-build-products
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

  nativeBuildInputs = [ git makeWrapper python311 stdenv.cc.cc.lib patchelf ];
  buildInputs = [ xrt ironvenv ];

  buildPhase = ''
    export HOME=$TMPDIR
    export XILINX_XRT="${xrt}/opt/xilinx/xrt"

    # utilities/export-kernels.py expects a writable venv at ./ironvenv with
    # its own bin/python. Copy the nix-provided venv and add a python symlink.
    rm -rf ironvenv
    cp -R ${ironvenv} ironvenv
    chmod -R +w ironvenv
    ln -sf ${python311}/bin/python ironvenv/bin/python

    export PEANO_INSTALL_DIR="$PWD/ironvenv/lib/python3.11/site-packages/llvm-aie"
    export PATH="$PEANO_INSTALL_DIR/bin:${xrt}/opt/xilinx/xrt/bin:$PATH"
    export PYTHONPATH="$PWD/ironvenv/lib/python3.11/site-packages:${xrt}/opt/xilinx/xrt/python''${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="${lib.makeLibraryPath [ stdenv.cc.cc.lib zlib xrt ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # The manylinux llvm-aie/mlir-aie wheels ship x86_64 executables that
    # hard-code /lib64/ld-linux-x86-64.so.2.  In the Nix sandbox /lib64 is
    # unavailable, so patch the interpreter to the stdenv linker and add the
    # C++ runtime + zlib to the RUNPATH without destroying the wheel's own
    # library search paths (e.g. mlir_aie.libs/libcrypto-...so.3).
    for dir in "$PEANO_INSTALL_DIR/bin" "$PWD/ironvenv/lib/python3.11/site-packages/mlir_aie/bin"; do
      if [ -d "$dir" ]; then
        for bin in "$dir/"*; do
          if [ -f "$bin" ] && [ -x "$bin" ]; then
            ${patchelf}/bin/patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" "$bin" 2> /dev/null || true
            ${patchelf}/bin/patchelf --add-rpath '${stdenv.cc.cc.lib}/lib:${zlib}/lib' "$bin" 2> /dev/null || true
          fi
        done
      fi
    done

    # Qwen2.5-3B's dx_attn set fails with:
    #   "dx_attn.py: this spec's q/k/v projections carry a bias and this design
    #    has no bias stream"
    # That is a recipe/design gap, not a packaging issue; skip it until the
    # open_kernels recipe covers attention bias in dx_attn.
    python utilities/export-kernels.py --force --jobs "''${NIX_BUILD_CORES:-4}" \
      ${lib.optionalString skipBert "--skip-bert"} \
      ${lib.optionalString (skipSpecs != "") "--skip-specs ${skipSpecs}"}
  '';

  installPhase = ''
    mkdir -p $out/share/oflm
    cp -r src/xclbins $out/share/oflm/
  '';

  # The open_npue BERT embedding sets need pyxrt and an NPU at build time.
  # Build with skipBert=false (and therefore __noChroot=true) only on a
  # machine that exposes /dev/accel* to the Nix builder.
  __noChroot = !skipBert;

  meta = {
    description = "OpenFlowLM open NPU kernel xclbins";
    platforms = [ "x86_64-linux" ];
  };
}

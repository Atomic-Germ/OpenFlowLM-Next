{
  source,
  lib,
  stdenv,
  fetchFromGitHub,
  pkgs,
  cmake,
  ninja,
  pkg-config,
  patchelf,
  autoPatchelfHook,
  abseil-cpp,
  boost,
  curl,
  ffmpeg,
  fftw,
  fftwFloat,
  fftwLongDouble,
  libdrm,
  libuuid,
  readline,
  ncurses,
  cargo,
  rustc,
  rustPlatform,
  xrt,
  makeWrapper,
  openflowlm-open-kernels,
  python3,
  oflmVersion ? "0.1.0",
  npuVersion ? "32.0.203.304",
}:

let
  # A flake source is a plain directory copy: git submodules are not checked
  # out into it, so third_party/tokenizers-cpp arrives empty even though the
  # gitlink is in the index. Fetch it (with its own submodules) at the pinned
  # revision the index records, and drop it in during postPatch.
  tokenizers-cpp = fetchFromGitHub {
    owner = "mlc-ai";
    repo = "tokenizers-cpp";
    rev = "c586c52f93f7b060753bd2388eb96a105cb7374d";
    hash = "sha256-r10QIeYnNaudFuHCdOWwTIHMRFqSf1p2ck33nmUEGj8=";
    fetchSubmodules = true;
  };

  # XRT's plugin loader resolves libxrt_driver_xdna next to libxrt_core.
  # Combine XRT with the amdxdna plugin so `oflm` works without relying on
  # the NixOS module's session variables.
  xrt-combined = pkgs.runCommand "xrt-combined" {} ''
    mkdir -p $out
    cp -rs ${xrt}/opt/xilinx/xrt/* $out/
    chmod -R u+w $out/lib
    ln -sf ${pkgs.xrt-plugin-amdxdna}/opt/xilinx/xrt/lib/libxrt_driver_xdna* $out/lib/
  '';

  in
stdenv.mkDerivation rec {
  pname = "openflowlm";
  version = oflmVersion;

  src = source;

  # The Rust tokenizer crate inside tokenizers-cpp has no Cargo.lock upstream.
  # We vendor its dependencies and inject the lock file at build time.
  cargoDeps = rustPlatform.importCargoLock {
    lockFile = ./tokenizers-cpp-cargo.lock;
  };
  cargoRoot = "third_party/tokenizers-cpp/rust";

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    patchelf
    autoPatchelfHook
    cargo
    rustc
    rustPlatform.cargoSetupHook
    makeWrapper
  ];

  buildInputs = [
    xrt
    abseil-cpp
    boost
    curl
    ffmpeg
    fftw
    fftwFloat
    fftwLongDouble
    libdrm
    libuuid
    readline
    ncurses
  ];

  propagatedBuildInputs = [ openflowlm-open-kernels ];

  cmakeFlags = [
    "-DOFLM_VERSION=${version}"
    "-DNPU_VERSION=${npuVersion}"
    "-DOFLM_BUILD_KERNELS=OFF"
    # The /etc/profile.d + /usr/bin plumbing is for an /opt prefix on a normal
    # distro; under Nix it would write outside $out (and fail in the sandbox).
    # NixOS puts the package on PATH itself.
    "-DOFLM_INSTALL_PATH_PLUMBING=OFF"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}"
    # sentencepiece defaults to FetchContent'ing abseil-cpp. Tell it to use
    # the system package instead; we still place our prefetched copy under
    # its expected third_party path so include/link paths resolve.
    "-DSPM_ABSL_PROVIDER=package"
  ];

  postPatch = ''
    rm -rf third_party/tokenizers-cpp
    mkdir -p third_party
    cp -r ${tokenizers-cpp} third_party/tokenizers-cpp
    chmod -R +w third_party/tokenizers-cpp
    cp ${./tokenizers-cpp-cargo.lock} third_party/tokenizers-cpp/rust/Cargo.lock

    # src/xclbins is git-ignored, so a flake source built from a working tree
    # carries whatever the developer last built locally and CMake would install
    # it alongside the packaged kernels. The kernels arrive from the
    # openflowlm-open-kernels input instead; start from an empty directory.
    rm -rf src/xclbins
  '';

  preConfigure = ''
    export XILINX_XRT="${xrt-combined}"
    export PKG_CONFIG_PATH="${xrt-combined}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  '';

  postInstall = ''
    # The kernels are a separate store path (they are built by a derivation that
    # needs the IRON toolchain). Point the engine's own "exe_dir/../share/oflm"
    # root at it, so `nix run .#oflm` finds the shipped sets with no environment
    # set up at all. The engine already searches ~/.config/oflm/xclbins on its
    # own, so a model added with `oflm add` is found alongside these.
    if [ -d ${openflowlm-open-kernels}/share/oflm/xclbins ]; then
      mkdir -p $out/share/oflm
      rm -rf $out/share/oflm/xclbins
      ln -s ${openflowlm-open-kernels}/share/oflm/xclbins $out/share/oflm/xclbins
    fi

    # The binary does not search $HOME for the model registry. Use a launcher
    # that defaults those variables to the user-level data `oflm add` writes,
    # falling back to nothing (the binary's own search finds the in-package
    # share/oflm). ":-" so an explicit export from the caller still wins.
    mv $out/bin/oflm $out/bin/.oflm-wrapped
    cat > $out/bin/oflm <<'EOF'
    #!/usr/bin/env bash
    export OFLM_CONFIG_PATH="''${OFLM_CONFIG_PATH:-$HOME/.config/oflm/model_list.json}"
    export OFLM_MODELINFO_PATH="''${OFLM_MODELINFO_PATH:-$HOME/.config/oflm/model_info.json}"
    # Deliberately NOT set OFLM_XCLBIN_PATH: it may point into the read-only
    # store, and oflm-add would then try to write model xclbins there.
    exec "@out@/bin/.oflm-wrapped" "$@"
    EOF
    chmod +x $out/bin/oflm
    substituteInPlace $out/bin/oflm --replace "@out@" "$out"

    wrapProgram $out/bin/.oflm-wrapped \
      --set-default XILINX_XRT "${xrt-combined}" \
      --prefix LD_LIBRARY_PATH : "${xrt-combined}/lib" \
      --prefix PATH : "${lib.makeBinPath [ xrt-combined python3 ]}"

    # oflm-test and q4nx-build are CMake-generated launchers that exec a bare
    # `python3`, and oflm-add.py starts with `#!/usr/bin/env python3`. None of
    # those resolve on a NixOS host, which has no /usr/bin/python3, so give each
    # one a Python to fall back on. It goes at the END of PATH: these tools need
    # third-party packages (openai, gguf, torch) that this package does not
    # vendor, and a venv or conda environment the user built should win.
    for prog in oflm-test q4nx-build; do
      if [ -x $out/bin/$prog ]; then
        wrapProgram $out/bin/$prog --suffix PATH : "${lib.makeBinPath [ python3 ]}"
      fi
    done
    if [ -f $out/share/oflm/oflm-add/oflm-add.py ]; then
      substituteInPlace $out/share/oflm/oflm-add/oflm-add.py \
        --replace "#!/usr/bin/env python3" "#!${python3}/bin/python3"
    fi
  '';

  doCheck = true;

  preCheck = ''
    # oflm_smoke runs `oflm list` which tries to create ~/.config/oflm on
    # first launch. Give it a writable home directory inside the sandbox.
    export HOME=$TMPDIR/home
    mkdir -p "$HOME/.config/oflm"
  '';

  meta = {
    description = "OpenFlowLM — NPU-offloaded LLM inference engine";
    homepage = "https://github.com/Atomic-Germ/OpenFlowLM";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "oflm";
  };
}

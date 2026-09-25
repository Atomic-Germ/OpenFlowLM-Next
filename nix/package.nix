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
  # The tokenizers-cpp submodule is listed in .gitmodules but its gitlink is
  # not in the index, so `fetchSubmodules` on the main repo does not populate
  # it. Fetch it explicitly here.
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
  '';

  preConfigure = ''
    export XILINX_XRT="${xrt-combined}"
    export PKG_CONFIG_PATH="${xrt-combined}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  '';

  postInstall = ''
    # The OFLM binary doesn't search $HOME for the model registry or xclbins.
    # Use a launcher script to point it at the user-level data that `oflm add`
    # writes.  This matches the desktop-launcher pattern used by lemonade and
    # other Nix-packaged tools that mix a read-only package with user state.
    mv $out/bin/oflm $out/bin/.oflm-wrapped
    cat > $out/bin/oflm <<'EOF'
    #!/usr/bin/env bash
    export OFLM_CONFIG_PATH="$HOME/.config/oflm/model_list.json"
    export OFLM_MODELINFO_PATH="$HOME/.config/oflm/model_info.json"
    export OFLM_XCLBIN_PATH="$HOME/.config/oflm"
    exec "@out@/bin/.oflm-wrapped" "$@"
    EOF
    chmod +x $out/bin/oflm
    substituteInPlace $out/bin/oflm --replace "@out@" "$out"

    wrapProgram $out/bin/.oflm-wrapped \
      --set-default XILINX_XRT "${xrt-combined}" \
      --prefix LD_LIBRARY_PATH : "${xrt-combined}/lib" \
      --prefix PATH : "${lib.makeBinPath [ xrt-combined python3 ]}"
  '';

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

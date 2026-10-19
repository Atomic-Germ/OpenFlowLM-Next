# Dev shell for building open NPU kernels.
# Reuses the same dependency source as the package derivation.
{
  pkgs ? null
, lib ? pkgs.lib
, stdenv ? pkgs.stdenv
, fetchurl ? pkgs.fetchurl
, python312 ? pkgs.python312
, python312Packages ? pkgs.python312Packages
, xrt ? pkgs.xrt
, git ? pkgs.git
, makeWrapper ? pkgs.makeWrapper
, unzip ? pkgs.unzip
, zlib ? pkgs.zlib
, patchelf ? pkgs.patchelf
}:

let
  env = pkgs.lib.callPackageWith {
    inherit lib stdenv fetchurl python312 python312Packages xrt git makeWrapper unzip zlib patchelf pkgs;
  } ./open-kernels-env.nix {};
in

pkgs.mkShell {
  name = "open-kernels-dev";

  nativeBuildInputs = env.nativeTools;
  buildInputs = env.runtimeLibs ++ [ env.ironvenv env.xrtCombined ];

  shellHook = ''
    export XILINX_XRT="${env.env.XILINX_XRT}"
    export PATH="$PWD/ironvenv/${env.pySite}/llvm-aie/bin:$PWD/ironvenv/${env.pySite}/mlir_aie/bin:$XILINX_XRT/bin:$PATH"
    export PYTHONPATH="$PWD/ironvenv/${env.pySite}:$XILINX_XRT/python''${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="${env.env.LD_LIBRARY_PATH}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # Materialize a writable venv in-repo so export scripts can write build
    # directories and runtime caches next to the source tree.
    if [ ! -d "$PWD/ironvenv" ]; then
      echo "[open-kernels-dev] materializing ironvenv..."
      cp -R ${env.ironvenv} "$PWD/ironvenv"
      chmod -R +w "$PWD/ironvenv"
    fi
    # The copied venv may contain absolute symlinks to the Python used at
    # build time; ensure the bin/python entry points at the shell's Python.
    ln -sf ${python312}/bin/python "$PWD/ironvenv/bin/python"
    if [ -e "$PWD/ironvenv/bin/python3" ]; then rm -f "$PWD/ironvenv/bin/python3"; fi
    ln -s python "$PWD/ironvenv/bin/python3"

    echo "[open-kernels-dev] XRT at $XILINX_XRT"
    echo "[open-kernels-dev] aiecc at $(which aiecc)"
    echo "[open-kernels-dev] Peano at $(which clang)"
  '';
}

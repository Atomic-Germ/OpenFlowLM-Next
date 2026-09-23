{ pkgs ? import <nixpkgs> { config.allowUnfree = true; }
, nix-amd-ai ? (builtins.getFlake (toString ./.)).inputs.nix-amd-ai
}:

let
  pkgs' = import pkgs.path {
    inherit (pkgs) system config;
    overlays = [ nix-amd-ai.overlays.default ];
  };

  env = pkgs'.lib.callPackageWith (
    {
      inherit (pkgs') lib stdenv fetchurl python312 python312Packages xrt git makeWrapper unzip zlib patchelf;
    }
  ) ./open-kernels-env.nix {};
in

pkgs'.mkShell {
  name = "oflm-dev";

  nativeBuildInputs = env.nativeTools;
  buildInputs = env.runtimeLibs ++ [ env.ironvenv ];

  shellHook = ''
    export XILINX_XRT="${env.env.XILINX_XRT}"
    export PATH="$PWD/ironvenv/${env.pySite}/llvm-aie/bin:$XILINX_XRT/bin:$PATH"
    export PYTHONPATH="$PWD/${env.pySite}:$XILINX_XRT/python''${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="${env.env.LD_LIBRARY_PATH}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # Materialize a writable venv in-repo so export scripts can write build
    # directories and runtime caches next to the source tree.
    if [ ! -d "$PWD/ironvenv" ]; then
      echo "[oflm-dev] materializing ironvenv..."
      cp -R ${env.ironvenv} "$PWD/ironvenv"
      chmod -R +w "$PWD/ironvenv"
      ln -sf ${pkgs'.python312}/bin/python "$PWD/ironvenv/bin/python"
    fi

    echo "[oflm-dev] XRT at $XILINX_XRT"
    echo "[oflm-dev] Peano at $(which clang)"
  '';
}

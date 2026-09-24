# q4nx-build packaged as a Nix application.
#
# The converter's runtime dependencies (torch, transformers, gguf, ...) currently
# drag in broken optional nixpkgs packages (fastapi/jupyter/inline-snapshot), so
# instead of building the full closure from nixpkgs we ship a wrapper that
# materializes a writable venv on first run and installs the pinned Python
# packages from PyPI. The interpreter and the q4nx-build source still come from
# Nix.
{
  lib,
  stdenv,
  python312,
  writeText,
  writeShellScript,
  zlib,
}:

let
  pyVersion = "3.12";

  # Minimal runtime dependencies for q4nx-build, taken from
  # utilities/q4nx-build/pyproject.toml, minus modelscope (not imported by q4nx
  # and marked insecure in nixpkgs).
  requirements = writeText "q4nx-build-requirements.txt" ''
    numpy
    torch
    einops
    matplotlib
    safetensors
    gguf
    huggingface-hub
    transformers
    accelerate
  '';

  setupRun = writeText "q4nx-build-setup-run.py" ''
    import os
    import subprocess
    import sys

    home = os.environ["Q4NX_BUILD_HOME"]
    src = os.environ["Q4NX_BUILD_SRC"]
    reqs = os.environ["Q4NX_BUILD_REQS"]
    py = os.environ["Q4NX_BUILD_PYTHON"]
    venv = os.path.join(home, "venv")
    site = os.path.join(venv, "lib", "python${pyVersion}", "site-packages")

    installed_marker = os.path.join(home, "venv-ready")
    if not os.path.exists(installed_marker):
        print(f"[q4nx-build] creating venv at {venv} ...")
        os.makedirs(home, exist_ok=True)
        subprocess.run([py, "-m", "venv", venv], check=True)
        pip = os.path.join(venv, "bin", "pip")
        subprocess.run([pip, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip, "install", "-r", reqs], check=True)
        with open(installed_marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
        print("[q4nx-build] venv ready")

    env = os.environ.copy()
    env["PATH"] = os.path.join(venv, "bin") + os.pathsep + env.get("PATH", "")
    # q4nx-build source lives in the Nix store; add it directly so the configs/
    # directory is found next to the q4nx package without an editable install.
    env["PYTHONPATH"] = src + os.pathsep + site + os.pathsep + env.get("PYTHONPATH", "")
    argv = [os.path.join(venv, "bin", "python"), "-m", "q4nx.cli"] + sys.argv[1:]
    os.execvpe(argv[0], argv, env)
  '';

  # Runtime wrapper: ensure HOME is sensible, then delegate to the Python shim.
  wrapper = writeShellScript "q4nx-build" ''
    if [ -z "$HOME" ] || [ "$HOME" = "/homeless-shelter" ]; then
      HOME="$(eval echo ~$USER)"
      if [ -z "$HOME" ] || [ "$HOME" = "/homeless-shelter" ]; then
        echo "q4nx-build: HOME is not set. Please set HOME or Q4NX_BUILD_HOME." >&2
        exit 1
      fi
    fi
    export Q4NX_BUILD_HOME="''${Q4NX_BUILD_HOME:-$HOME/.cache/oflm/q4nx-build}"
    export Q4NX_BUILD_SRC="@out@/share/q4nx-build"
    export Q4NX_BUILD_REQS="@out@/share/q4nx-build/requirements.txt"
    export Q4NX_BUILD_PYTHON="${python312}/bin/python"
    # PyPI manylinux wheels (numpy, torch, gguf, matplotlib, ...) need
    # libstdc++.so.6 and libz.so.1 which the venv does not provide itself.
    export LD_LIBRARY_PATH="${lib.makeLibraryPath [ stdenv.cc.cc.lib zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    exec ${python312}/bin/python @out@/share/q4nx-build/setup_run.py "$@"
  '';

in
stdenv.mkDerivation {
  pname = "q4nx-build";
  version = "0.3.0";

  src = ../utilities/q4nx-build;

  installPhase = ''
    runHook preInstall

    mkdir -p $out/share/q4nx-build
    cp -r $src/. $out/share/q4nx-build/
    cp ${requirements} $out/share/q4nx-build/requirements.txt
    cp ${setupRun} $out/share/q4nx-build/setup_run.py

    mkdir -p $out/bin
    sed -e "s|@out@|$out|g" ${wrapper} > $out/bin/q4nx-build
    chmod +x $out/bin/q4nx-build

    runHook postInstall
  '';

  # Nix does not pick this up automatically because LD_LIBRARY_PATH is set at
  # runtime, not in the RPATH of the wrapper. Add the GCC lib path here so that
  # `nix run` and `nix develop` behave the same without relying on the user's
  # environment.
  propagatedBuildInputs = [ stdenv.cc.cc.lib ];

  meta = {
    description = "Convert GGUF or HF models into OFLM's Q4NX format";
    license = lib.licenses.mit;
    mainProgram = "q4nx-build";
    platforms = [ "x86_64-linux" ];
  };
}

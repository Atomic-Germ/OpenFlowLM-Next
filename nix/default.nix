# OpenFlowLM as a Nix derivation
#
# It packages the release tarball (openflowlm-<version>-Linux.tar.gz, built by
# `cpack -G TGZ` in the release workflow) rather than compiling the engine
# inside Nix. That is a deliberate first cut, and the reason is XRT: the AMD NPU
# runtime is not in nixpkgs, so a from-source build needs it vendored as a
# fixed-output derivation with a hash nobody here can produce or refresh on
# demand. The binary distribution already has its ABI fixed by the build host,
# and its RUNPATH is $ORIGIN/../lib64, so it drops into the store unmodified.
#
# The version and the tarball hash come from pin.json, which the release
# workflow regenerates for the version it just published. See nix/README.md.
{ lib
, stdenvNoCC
, fetchurl
, makeWrapper
, patchelf
, python3
, version
, sha256
}:

stdenvNoCC.mkDerivation {
  pname = "openflowlm";
  inherit version;

  src = fetchurl {
    url = "https://github.com/Atomic-Germ/OpenFlowLM-Next/releases/download/"
         + "v${version}/openflowlm-${version}-Linux.tar.gz";
    inherit sha256;
  };

  # The top directory CPack's TGZ generator writes, verified against a real
  # package: openflowlm-<version>-Linux/{opt,etc,usr}.
  sourceRoot = "openflowlm-${version}-Linux";

  nativeBuildInputs = [ makeWrapper patchelf ];

  dontConfigure = true;

  installPhase = ''
    runHook preInstall

    mkdir -p "$out"
    cp -r opt/openflowlm/. "$out/"

    # opt/openflowlm only. The package also carries /etc/profile.d/openflowlm.sh
    # and the /usr/bin/oflm symlink, both absolute, and both the package
    # manager's business in Nix: `environment.systemPackages` puts the wrapper
    # on PATH, and there is no profile.d to drop into.

    # The bundled utility launchers (oflm-test, q4nx-build) are generated at
    # install time with the build-time prefix baked in, so they would point at
    # /opt/openflowlm here. The engine itself needs no patching: its RUNPATH is
    # $ORIGIN/../lib64, which survives the store move and lands on lib64/.
    for launcher in "$out/bin/oflm-test" "$out/bin/q4nx-build"; do
      if [ -e "$launcher" ]; then
        substituteInPlace "$launcher" --replace /opt/openflowlm "$out"
        wrapProgram "$launcher" --prefix PATH : ${lib.makeBinPath [ python3 ]}
      fi
    done

    runHook postInstall
  '';

  # The two things a Nix package has to get right that an RPM does not: it has
  # to run from the store, and it has to not need /opt. This is the check.
  doCheck = true;
  checkPhase = ''
    runHook preCheck
    export HOME="$TMPDIR/home"
    export XDG_CONFIG_HOME="$TMPDIR/home/.config"
    mkdir -p "$XDG_CONFIG_HOME"
    echo "--- oflm --version"
    "$out/bin/oflm" --version
    echo "--- oflm list (proves the engine libs, the registry and the xclbin"
    echo "--- roots all resolve from the store)"
    "$out/bin/oflm" list
    runHook postCheck
  '';

  meta = with lib; {
    description = "OpenFlowLM - LLM inference with dense GEMM offloaded to the AMD NPU2";
    homepage = "https://github.com/Atomic-Germ/OpenFlowLM-Next";
    license = licenses.mit;
    # Linux only, and not because of the build: the engine links the AMD NPU
    # runtime, and the only configuration of it that exists is a Linux XRT.
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}

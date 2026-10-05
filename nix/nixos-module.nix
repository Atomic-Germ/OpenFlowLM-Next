# NixOS module for OpenFlowLM.
# Import it through the flake's `nixosModules.default`/`nixosModules.openflowlm`,
# which add the overlay these option defaults rely on and pull in the nix-amd-ai
# NPU module.  It wires:
#   - the openflowlm engine package onto PATH
#   - the open NPU kernel xclbins into the engine package's share/oflm/xclbins
#     (the engine's own default search root)
#   - hardware.amd-npu.enable (a mkDefault, so a host's own setting wins)
{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.programs.openflowlm;
  inherit (lib) mkEnableOption mkIf mkOption types;

  # Each kernel package contributes <pkg>/share/oflm/xclbins. Merge them into
  # one tree, then link that tree into the engine package -- the engine looks at
  # <exe_dir>/../share/oflm before any environment variable is set, so this is
  # what makes the system-wide kernels work for every user with no profile edit.
  kernelXclbins = pkgs.symlinkJoin {
    name = "openflowlm-xclbins";
    paths = map (p: p + "/share/oflm/xclbins") ([ cfg.kernelsPackage ] ++ cfg.extraXclbinPackages);
  };

  engine = cfg.package.overrideAttrs (old: {
    postInstall = (old.postInstall or "") + ''
      mkdir -p $out/share/oflm
      ln -sfn ${kernelXclbins} $out/share/oflm/xclbins
    '';
  });
in
{
  options.programs.openflowlm = {
    enable = mkEnableOption "OpenFlowLM NPU-offloaded LLM inference engine";

    package = mkOption {
      type = types.package;
      default = pkgs.oflm;
      defaultText = lib.literalExpression "pkgs.oflm";
      description = "The OpenFlowLM engine package to install.";
    };

    kernelsPackage = mkOption {
      type = types.package;
      default = pkgs.openflowlm-open-kernels;
      defaultText = lib.literalExpression "pkgs.openflowlm-open-kernels";
      description = "The open NPU kernel xclbin package to install alongside the engine.";
    };

    extraXclbinPackages = mkOption {
      type = types.listOf types.package;
      default = [];
      description = ''
        Additional packages whose share/oflm/xclbins directories should be made
        available to the engine.  Use this for kernels that cannot be built in
        the Nix sandbox, such as the BERT embedding sets; build them
        imperatively with `nix develop .#open-kernels` and point this option
        at the result.
      '';
    };

    enableNPU = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether to enable the AMD XDNA2 NPU runtime from nix-amd-ai.  Default
        is true because OpenFlowLM is designed to run on the NPU.  Set to
        false on build/CI hosts that have no NPU, so the engine and kernels
        can still be installed and built without loading amdxdna or the XRT
        plugin.
      '';
    };
  };

  config = mkIf cfg.enable {
    # OpenFlowLM needs the AMD XDNA2 NPU stack.  If the host already imports
    # nix-amd-ai or sets hardware.amd-npu.enable, this default has no effect.
    hardware.amd-npu.enable = lib.mkDefault cfg.enableNPU;

    environment.systemPackages = [ engine ];
  };
}

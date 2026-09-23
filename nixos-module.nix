# NixOS module for OpenFlowLM.
# Import this via your flake's nixosConfigurations using
#   inputs.openflowlm.nixosModules.default
# It wires:
#   - the openflowlm engine package onto PATH
#   - the open NPU kernel xclbins into /run/current-system/sw/share/oflm/xclbins
#     (matching the engine's default search path)
#   - the nix-amd-ai XRT + amdxdna plugin when the NPU is enabled
{
  config,
  lib,
  pkgs,
  self,
  ...
}:

let
  cfg = config.programs.openflowlm;
  inherit (lib) mkEnableOption mkIf mkOption types;
in
{
  # nix-amd-ai owns the NPU kernel module, udev rules, PAM memlock limits, and
  # XRT + amdxdna plugin wiring.  Import it unconditionally: NixOS modules are
  # idempotent, so hosts that already import it are unaffected, while hosts
  # that don't can enable the NPU via programs.openflowlm.enableNPU below.
  imports = [ self.inputs.nix-amd-ai.nixosModules.default ];

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

    environment.systemPackages = [ cfg.package cfg.kernelsPackage ] ++ cfg.extraXclbinPackages;
  };
}

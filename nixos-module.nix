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
  options.programs.openflowlm = {
    enable = mkEnableOption "OpenFlowLM NPU-offloaded LLM inference engine";

    package = mkOption {
      type = types.package;
      default = self.packages.${pkgs.system}.oflm;
      defaultText = lib.literalExpression "self.packages.\${pkgs.system}.oflm";
      description = "The OpenFlowLM engine package to install.";
    };

    kernelsPackage = mkOption {
      type = types.package;
      default = self.packages.${pkgs.system}.openflowlm-open-kernels;
      defaultText = lib.literalExpression "self.packages.\${pkgs.system}.openflowlm-open-kernels";
      description = "The open NPU kernel xclbin package to install alongside the engine.";
    };

    enableNPU = mkOption {
      type = types.bool;
      default = true;
      description = "Whether to enable the AMD XDNA2 NPU runtime from nix-amd-ai.";
    };
  };

  config = mkIf cfg.enable {
    # Pull in the nix-amd-ai NPU module if requested.  It loads the amdxdna
    # kernel module, sets up udev rules for /dev/accel*, and wires XRT with
    # the amdxdna plugin so xrt-smi / pyxrt can enumerate the device.
    imports = mkIf cfg.enableNPU [ self.inputs.nix-amd-ai.nixosModules.default ];

    programs.openflowlm.enableNPU = lib.mkDefault cfg.enableNPU;

    # The kernel package installs xclbins under $out/share/oflm/xclbins; placing it
    # on systemPackages makes them available at /run/current-system/sw/share/oflm/xclbins,
    # matching how nix-amd-ai exposes FastFlowLM's bundled xclbins.
    environment.systemPackages = [ cfg.package cfg.kernelsPackage ];
  };
}

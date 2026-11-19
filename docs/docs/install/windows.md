---
title: "Windows"
description: "Install OpenFlowLM on Windows 11 with the MSI installer and AMD Ryzen AI NPU drivers."
icon: "desktop_windows"
weight: 10
aliases: ["/docs/install_win/"]
---

Install OpenFlowLM on Windows 11 (`x64`, 24H2 recommended) with an AMD Ryzen AI XDNA2 processor, NPU driver `32.0.203.304+`, and at least `16 GB` of RAM (`32 GB+` recommended).

## 1. Hardware & NPU Driver Compatibility {#step-1-hardware-and-driver}

| Processor / Driver | Requirement | NPU | Status |
|---|---|---|---|
| Ryzen AI 300 / 400 / Max 300 / Z2 Extreme | AMD Ryzen AI NPU driver `32.0.203.304+` ([AMD Support](https://www.amd.com/en/support)) | XDNA2 | Supported |
| Older NPU drivers | `< 32.0.203.304` | XDNA2 | Unsupported |
| Ryzen 7040 / 8040 / 200-series | Phoenix, Hawk Point | XDNA1 | Unsupported |

Check your installed driver version in **Task Manager -> Performance -> NPU** or **Device Manager -> Neural processors -> NPU Compute Accelerator Device**.

## 2. Run the MSI Installer {#step-2-run-the-msi-installer}

Download and run [`oflm-setup.msi`](https://github.com/Atomic-Germ/OpenFlowLM-Next/releases/latest/download/oflm-setup.msi) (see [GitHub Releases](https://github.com/Atomic-Germ/OpenFlowLM-Next/releases) for changelogs). If Windows SmartScreen prompts you, click **More info -> Run anyway**.

For silent installation in PowerShell:

```powershell
.\oflm-setup.msi /qn
```

To compile from source on Windows 11, see [Build System](/docs/contributing/build/) and [`src/WinSetup.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/src/WinSetup.md).

## 3. Validate and Run {#step-3-validate-and-run}

Open PowerShell, verify your NPU setup, and install a model:

```powershell
oflm validate
oflm add unsloth/Qwen3.5-4B-GGUF
oflm run qwen3.5:4b
```

Monitor live utilization in **Task Manager -> Performance -> NPU**. Inside `oflm run`, type `/verbose` for token/s metrics and `/bye` to exit.

> [!NOTE]
> Models default to `C:\Users\<USER>\.oflm\models\` and user overrides live in `C:\Users\<USER>\.oflm\settings.json`. Set `OFLM_MODEL_PATH` to relocate storage, pass `--modelscope` if Hugging Face is unreachable, and run `oflm config show` to inspect active paths.

## NPU Power Mode

OpenFlowLM defaults to `performance` power mode. Override it per run with `--pmode` (`default`, `powersaver`, `balanced`, `performance`, `turbo`) or persist a default with [`oflm config`](/docs/instructions/config/):

```powershell
oflm run qwen3.5:4b --pmode balanced
oflm serve qwen3.5:4b --pmode balanced
oflm config set --config-key pmode --config-value turbo
```

> [!WARNING]
> `powersaver` and `balanced` reduce NPU clocks and token throughput.

Continue to [CLI Usage](/docs/instructions/cli/) or [Server Mode](/docs/instructions/server/).

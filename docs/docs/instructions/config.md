---
title: "Configuration"
description: "Configure model storage paths, context length, power mode, server port, and per-model settings."
icon: "tune"
weight: 20
---

OpenFlowLM separates built-in defaults from your personal configuration so software upgrades preserve your preferences and models registered via [`oflm add`](/docs/instructions/cli/#add-a-model-oflm-add):

| File | Owner | Upgrade Behavior |
|---|---|---|
| `model_list.json` (built-in) | OpenFlowLM | Replaced on upgrade |
| `settings.json` (`~/.config/oflm/settings.json` on Linux, `%USERPROFILE%\.oflm\settings.json` on Windows) | User | Preserved across upgrades |

`settings.json` overlays `model_list.json` at startup, storing only the settings and custom models you add or change.

{{< wip issue="240" title="Built-in Registry Migration" >}}`oflm add <hf-repo-or-file>` already registers direct GGUF models in `settings.json`. Built-in `oflm pull <tag>` shortcuts in `model_list.json` are migrating from legacy `.q4nx` archives to upstream GGUFs under #240, and speculative decoding (`--speculative` / `OFLM_SPECULATIVE`) is in progress under #189 and #240.{{< /wip >}}

## Precedence, Storage Paths & Server Bindings

Settings resolve from lowest to highest priority: `model_list.json` -> `settings.json` -> `OFLM_*` environment variables -> [CLI Usage](/docs/instructions/cli/) flags -> per-request HTTP fields ([Server Mode](/docs/instructions/server/)).

| Variable / Flag | Scope | Purpose |
|---|---|---|
| `OFLM_MODEL_PATH` | Global | Parent folder for downloaded weights (stored in `$OFLM_MODEL_PATH/models`) |
| `OFLM_SETTINGS_PATH` | Global | Custom location for your user `settings.json` file |
| `OFLM_SERVE_PORT` / `--port` | Server | Default HTTP port (`52625`) or per-run server port override |
| `--host <addr>` | Server | Bind address for `oflm serve` (`127.0.0.1` default; Docker `.env` uses `OFLM_BIND_ADDRESS`) |
| `OFLM_CONFIG_PATH` / `OFLM_MODELINFO_PATH` | Global | Custom paths to built-in `model_list.json` and `model_info.json` |
| `OFLM_XCLBIN_PATH` | Global | Custom parent directory containing bundled NPU acceleration files |
| `OFLM_OPEN_PREFILL_MODE` | Runtime | Prompt processing memory/speed preset (`fast`, `lean`, or `bfp16` on Granite 4.2 3B) |
| `OFLM_OPEN_PREFILL_PRECISION` | Runtime | Prompt processing precision preset (`paired`, `three_term`, `weight_high`, `single`) |
| `OFLM_SPECULATIVE` | Runtime | Enable or disable speculative draft decoding (`0` or `1`) |

## `oflm config` Commands

```bash
# Inspect active storage paths and effective settings
oflm config show
oflm config show --config-model qwen3.8:27b
oflm config show --json

# Save or clear global and per-model overrides
oflm config set --config-key pmode --config-value balanced
oflm config set --config-model llama3.1:8b --config-key ctx_len --config-value 32768
oflm config set --config-model qwen3.8:27b --config-key prefill_mode --config-value lean
oflm config unset --config-key pmode

# Interactive menu, full-screen browser, or $EDITOR
oflm config
oflm config browse
oflm config edit

# Reconcile overrides after an OpenFlowLM upgrade
oflm config review
oflm config review --accept-shipped
oflm config review --keep-all
```

> [!TIP]
> Interactive menus accept arrow keys and Vim keys (`j`/`k` move, `l`/`Enter` select, `h`/`Esc` back, `q` quit). Delete `settings.json` at any time to return to built-in defaults.

## Persistent Settings Reference

| Key | CLI Flag | Env Var | Scope | Accepted Values |
|---|---|---|---|---|
| `pmode` | `--pmode` | *(none)* | Global | `default`, `powersaver`, `balanced`, `performance` (default), `turbo` |
| `ctx_len` | `--ctx-len` (`-c`) | *(none)* | Per-model | `int >= 512` (presets: `2048` to `32768`) |
| `prefill_chunk_len` | `--prefill-chunk-len` | *(none)* | Per-model | `int >= 512` (presets: `1024` to `8192`) |
| `prefill_mode` | `--prefill-mode` | `OFLM_OPEN_PREFILL_MODE` | Per-model | `fast` (default), `lean` (lower RAM, ~15% slower), or `bfp16` (Granite 4.2 3B) |
| `prefill_precision` | `--prefill-precision` | `OFLM_OPEN_PREFILL_PRECISION` | Per-model | `paired` (default), `three_term`, `weight_high`, `single` |
| `think` | *(none)* | *(none)* | Per-model | `true`, `false` (default reasoning output for thinking models) |
| `speculative` | `--speculative` | `OFLM_SPECULATIVE` | Per-model | `true`, `false` ({{< b k="wip" issue="189" t="WIP #189" >}} {{< b k="wip" issue="240" t="WIP #240" >}}) |

## Legacy Migration

If an older `~/.config/oflm/model_list.json` exists when `settings.json` has not been created yet, OpenFlowLM automatically imports your custom models and modified settings into `settings.json` on startup and saves a backup at `model_list.json.bak`.

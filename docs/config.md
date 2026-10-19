# Configuration

OpenFlowLM uses a two-file configuration system:

| File | Owner | Overwritten on release? |
|------|-------|------------------------|
| `model_list.json` (shipped) | OpenFlowLM | Yes |
| `~/.config/oflm/settings.json` | You | **Never** |

## How it works

The shipped `model_list.json` contains the model registry and default settings.
Your `settings.json` is a **sparse overlay** — only the settings you changed.
When oflm starts, your settings are deep-merged on top of the shipped defaults.

## Precedence (lowest to highest)

1. Shipped defaults (in `model_list.json`)
2. Your `settings.json`
3. `OFLM_*` environment variables
4. Command-line flags (`--ctx-len`, `--pmode`, etc.)
5. Per-request API fields (serve mode)

## Commands

### Show effective settings

```bash
oflm config show                          # Global settings
oflm config show --config-model qwen3.8:27b   # One model's settings
oflm config show --json                   # Machine-readable
```

### Set a preference

```bash
oflm config set --config-key pmode --config-value balanced           # Global
oflm config set --config-model qwen3.8:27b --config-key prefill_mode --config-value lean
oflm config set --config-model qwen3.8:27b --config-key prefill_mode  # Interactive picker
```

### Remove a preference (revert to shipped default)

```bash
oflm config unset --config-key pmode
oflm config unset --config-model qwen3.8:27b --config-key prefill_mode
```

### Edit the file directly

```bash
oflm config edit    # Opens ~/.config/oflm/settings.json in $EDITOR
```

### Interactive guided flow

```bash
oflm config         # Interactive: pick model → setting → value
```

### Full-screen TUI

```bash
oflm config browse  # Browse and edit all settings in a full-screen interface
```

### Reconcile after updates

When a release changes shipped defaults that you've overridden:

```bash
oflm config review                        # Interactive: keep/remove/skip each override
oflm config review --accept-shipped       # Remove all redundant overrides
oflm config review --keep-all             # Keep all, just update version
```

## Settings reference

| Setting | Type | Options | Scope | Description |
|---------|------|---------|-------|-------------|
| `prefill_mode` | enum | `fast`, `lean` | per-model | Memory/speed trade-off for prefill |
| `pmode` | enum | `default`, `powersaver`, `balanced`, `performance`, `turbo` | global | NPU power mode |
| `ctx_len` | int | min 512 | per-model | Context window length |
| `prefill_chunk_len` | int | min 512 | per-model | Maximum prefill chunk length |
| `think` | bool | `true`, `false` | per-model | Enable reasoning/thinking mode |

## Examples

### Lower memory usage for a specific model

```bash
oflm config set --config-model qwen3.8:27b --config-key prefill_mode --config-value lean
```

### Change global power mode

```bash
oflm config set --config-key pmode --config-value balanced
```

### See what you've changed

```bash
oflm config show
# pmode    balanced    (user)
# ...
```

### Revert everything to defaults

```bash
rm ~/.config/oflm/settings.json
```

## Migration from older versions

If you previously used `oflm add` (before this feature), your old
`~/.config/oflm/model_list.json` will be migrated automatically on first run:

- User-added models move to `settings.json`
- Modified shipped entries are preserved as overrides
- Stale copies are dropped (backup kept as `model_list.json.bak`)

## File locations

| Platform | settings.json path |
|----------|-------------------|
| Linux | `~/.config/oflm/settings.json` |
| Windows | `%USERPROFILE%\.oflm\settings.json` |
| Custom | Set `OFLM_SETTINGS_PATH` |

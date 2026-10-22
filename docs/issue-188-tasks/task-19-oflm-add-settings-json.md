# Task 19: oflm-add — Settings.json Path + register() Rewrite

**Phase:** 7 (oflm add Changes)
**Depends on:** None (independent Python task)
**Files to modify:** `utilities/oflm-add/oflm_add/__init__.py`
**Estimated complexity:** Medium (~100 lines changed)

## Context

Currently `oflm add` writes to `~/.config/oflm/model_list.json` (a full copy of the shipped registry + user models). This breaks the two-file rule. This task changes it to write to `~/.config/oflm/settings.json` under the `models` section.

Read parent plan **§Phase 7** in `docs/issue-188-implementation-plan.md`.

## What to do

1. Rename `user_registry_path()` → `user_settings_path()` — returns settings.json path
2. Rewrite `register()` to write to settings.json → models section
3. Remove the deep-copy of system registry on first add

## Code changes in `utilities/oflm-add/oflm_add/__init__.py`

### 1. Replace `user_registry_path()` (around line 346-352)

```python
def user_settings_path(cli_config: str | None = None) -> Path:
    """Path to the user's settings.json (overlay file, never overwritten by releases)."""
    if cli_config:
        return Path(cli_config)
    env_path = os.environ.get("OFLM_SETTINGS_PATH")
    if env_path:
        return Path(env_path)
    # Default: sibling of models directory
    models_root = models_root_dir()
    return models_root / "settings.json"


# Backward compat alias
def user_registry_path(cli_config: str | None = None) -> Path:
    """Deprecated alias for user_settings_path()."""
    return user_settings_path(cli_config)
```

### 2. Rewrite `register()` (around line 1150-1158)

```python
def register(
    entry: dict,
    settings_path: Path,
    system_registry: dict,
    force: bool = False,
    name_override: str | None = None,
) -> str:
    """Register a model in the user's settings.json (not model_list.json).

    Returns the tag the model was registered under.
    Raises SystemExit on collision (without force/name_override).
    """
    # Load user settings (create if missing)
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
    else:
        settings = {
            "version": 1,
            "defaults": {},
            "settings": {},
            "models": {},
        }

    # Determine tag
    model_type = entry.get("type", "unknown")
    size = entry.get("size", "unknown")
    tag = name_override or f"{model_type}:{size}"

    if ":" in tag:
        model_type, size = tag.split(":", 1)

    # Collision check: shipped registry
    if (model_type in system_registry.get("models", {}) and
        size in system_registry["models"][model_type]):
        if not name_override:
            raise SystemExit(
                f"Model {tag} already exists in the shipped registry. "
                f"Use --name to install under a different name."
            )

    # Collision check: user settings
    if (model_type in settings.get("models", {}) and
        size in settings["models"][model_type]):
        if not force:
            raise SystemExit(
                f"Model {tag} is already installed. Use --force to overwrite."
            )

    # Write complete entry to settings.models
    settings.setdefault("models", {}).setdefault(model_type, {})[size] = entry

    # Ensure parent directory exists
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # Save atomically
    save_json_atomic(settings_path, settings)

    return tag


def save_json_atomic(path: Path, data: dict) -> None:
    """Save JSON atomically (temp file + rename)."""
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except:
        try:
            os.unlink(tmp)
        except:
            pass
        raise
```

### 3. Update `main()` to use new function

Find where `register()` is called (around line 2210) and update:

```python
# Old:
# register(entry, user_registry_path(args.config), system_registry)

# New:
settings_path = user_settings_path(args.config)
tag = register(
    entry,
    settings_path,
    system_registry,
    force=args.force,
    name_override=args.name,
)
print(f"Model {tag} installed to {settings_path}")
print(f"Run with: oflm run {tag}")
```

## Test

```bash
cd utilities/oflm-add
python3 -m pytest tests/ -x -q 2>&1 | tail -10

# Manual test
python3 -c "
import sys; sys.path.insert(0, '.')
from oflm_add import user_settings_path, register
import json, tempfile, os
from pathlib import Path

# Create temp settings
tmp = Path(tempfile.mkdtemp())
sp = tmp / 'settings.json'

# Create fake system registry
sys_reg = {'models': {'qwen3.8': {'27b': {'name': 'qwen3.8-27b'}}}}

# Register a user model
entry = {'type': 'my-model', 'size': '1b', 'name': 'my-model-1b', 'url': 'https://example.com'}
tag = register(entry, sp, sys_reg)
print(f'Registered: {tag}')

# Verify settings.json
with open(sp) as f:
    s = json.load(f)
assert 'my-model' in s['models'], 'Model not in settings.json'
assert s['models']['my-model']['1b']['url'] == 'https://example.com'
print('OK: settings.json has the model')

# Verify collision with shipped
try:
    register({'type': 'qwen3.8', 'size': '27b', 'name': 'x'}, sp, sys_reg)
    print('FAIL: should have raised')
except SystemExit as e:
    print(f'OK: collision detected: {e}')
"
```

## Success criteria

- [ ] `user_settings_path()` function added (returns settings.json path)
- [ ] `user_registry_path()` kept as backward-compat alias
- [ ] `register()` rewritten to write to settings.json → models section
- [ ] `register()` does NOT deep-copy system registry
- [ ] `save_json_atomic()` helper added
- [ ] Collision detection: shipped registry → error with --name suggestion
- [ ] Collision detection: user model → error with --force suggestion
- [ ] `main()` updated to use new register() signature
- [ ] Existing oflm-add tests still pass (or are updated)
- [ ] Manual test passes

#!/usr/bin/env python3
"""Inspect a converted model against the open kernel families available locally.

This is diagnostic, not an allowlist. It reports actual container requirements,
known-good/curated status, and family/shape candidates as separate facts. A model
does not need a model_list.json entry or an exact spec_hash to be considered.

The first implementation targets a local converted model directory. Recipe
discovery is optional; without a checkout the command still reports model
metadata and exact manifest matches, but cannot claim recipe validation.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import struct
import sys
import urllib.parse
from pathlib import Path
from typing import Any


# Kernel recipe family -> OpenFlowLM runtime family. This is family-level
# routing metadata, not a per-model allowlist. Names of finetunes never appear
# here. Keep it aligned with the engine's family map and q4nx-build's
# model_type-to-family mapping.
RUNTIME_FAMILY = {
    "qwen35": "qwen3.5",
    "qwen36moe": "qwen3.6-moe",
    "qwen3": "qwen3",
    "llama3": "llama3",
    "qwen2": "qwen2",
    "gemma3": "gemma3",
    "granite": "granite",
    "phi3": "phi4",
    "lfm2": "lfm2",
    "gptoss": "gpt-oss",
}
MODEL_TYPE_RUNTIME_FAMILY = {
    "qwen3_vl": "qwen3vl", "qwen3_vl_text": "qwen3vl",
    "qwen2_5_vl": "qwen2.5vl", "qwen2_5_vl_text": "qwen2.5vl",
}


def _recipe_root(explicit: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("OPEN_KERNELS_DIR")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates.extend(p / "open_kernels" for p in here.parents)
    candidates.append(Path.cwd() / "open_kernels")
    for path in candidates:
        if (path / "recipes" / "spec.py").is_file():
            return path.resolve()
    return None


def _load_model(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any], int | None]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise ValueError(f"{model_dir}: missing config.json")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    q4nx_path = model_dir / "model.q4nx"
    if not q4nx_path.is_file():
        raise ValueError(f"{model_dir}: missing model.q4nx")
    with q4nx_path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{q4nx_path}: truncated safetensors header length")
        size = struct.unpack("<Q", raw_len)[0]
        header_bytes = f.read(size)
        if len(header_bytes) != size:
            raise ValueError(f"{q4nx_path}: truncated safetensors header")
    header = json.loads(header_bytes)

    real_vocab = None
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.is_file():
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        ids = list(tokenizer.get("model", {}).get("vocab", {}).values())
        ids.extend(t["id"] for t in tokenizer.get("added_tokens", []) if "id" in t)
        if ids:
            real_vocab = max(int(i) for i in ids) + 1
    return config, header, real_vocab


def _model_spec(model_dir: Path, recipe_root: Path) -> tuple[Any, dict[str, Any]]:
    """Use the recipes' own container/config derivation; do not clone it here."""
    root = str(recipe_root)
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        from recipes.load import spec_from_model_dir
        from recipes.families import for_spec
        from recipes.manifest import manifest

        spec = spec_from_model_dir(model_dir)
        family_recipe = for_spec(spec)
        # Permit catalogue-only off-grid points for diagnosis, but structural
        # refusals still raise. The environment is restored before returning.
        old = os.environ.get("OPEN_KERNELS_UNVALIDATED")
        os.environ["OPEN_KERNELS_UNVALIDATED"] = "1"
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                derived = manifest(spec)
        finally:
            if old is None:
                os.environ.pop("OPEN_KERNELS_UNVALIDATED", None)
            else:
                os.environ["OPEN_KERNELS_UNVALIDATED"] = old
        note = "catalogue-unvalidated" if "UNVALIDATED point allowed" in stderr.getvalue() else "validated"
        return spec, {"manifest": derived, "recipe_status": note,
                      "recipe_module": family_recipe.__name__}
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def _manifest_paths(roots: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        if (root / "manifest.json").is_file():
            found.add(root / "manifest.json")
        found.update(root.glob("*/open_kernels/manifest.json"))
        found.update(root.glob("*/manifest.json"))
    return sorted(found)


def _without(d: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in keys}


def _kernel_contract(m: dict[str, Any]) -> dict[str, Any]:
    """Fields that describe the family/shape kernel contract, not model identity.

    `extra`, `real_vocab`, spec_hash and build_key are deliberately excluded.
    real_vocab is a model/tokenizer property; the padded `vocab` and head pool
    geometry remain in the contract. Runtime manifest adaptation is reported
    separately rather than falsely equating provenance with eligibility.
    """
    spec = _without(copy.deepcopy(m.get("spec", {})), "extra", "real_vocab")
    layout = copy.deepcopy(m.get("layout", {}))
    layout.pop("real_vocab", None)
    return {
        "family": m.get("family"),
        "spec": spec,
        "hf_config_check": m.get("hf_config_check", {}),
        "hf_config_defaults": m.get("hf_config_defaults", {}),
        "layout": layout,
        "layers": m.get("layers", []),
        "contexts": m.get("contexts", {}),
        "kernels": m.get("kernels", {}),
        "layer_types": m.get("layer_types", {}),
        "tail": m.get("tail", []),
        "globals": m.get("globals", {}),
        "builds": m.get("builds", {}),
        "pack": m.get("pack", {}),
    }


def _diff_paths(a: Any, b: Any, prefix: str = "") -> list[str]:
    if type(a) is not type(b):
        return [prefix or "<root>"]
    if isinstance(a, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            p = f"{prefix}.{key}" if prefix else key
            if key not in a or key not in b:
                out.append(p)
            else:
                out.extend(_diff_paths(a[key], b[key], p))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [prefix]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_diff_paths(x, y, f"{prefix}[{i}]"))
        return out
    return [] if a == b else [prefix]


def _load_manifests(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("manifest_version"):
                out.append((path.parent, value))
        except (OSError, ValueError):
            continue
    return out


def _config_manifest_differences(config: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Mirror the manifest's fail-closed config checks without importing C++."""
    expected = manifest.get("hf_config_check", {})
    defaults = manifest.get("hf_config_defaults", {})
    differences: list[str] = []
    for key, want in expected.items():
        if key == "model_type":
            if config.get(key) not in want:
                differences.append("model_type")
            continue
        if key == "layer_types":
            got = config.get("layer_types")
            if got is None and "full_attention_interval" in config and "num_hidden_layers" in config:
                interval = int(config["full_attention_interval"])
                got = (["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                        for i in range(int(config["num_hidden_layers"]))] if interval > 0 else None)
            if got != want:
                differences.append("layer_types")
            continue
        got = config.get(key, defaults.get(key))
        if key not in config and key not in defaults or got != want:
            differences.append(key)
    return differences


def _inspect_without_recipes(result: dict[str, Any], config: dict[str, Any], manifests,
                             model_list: Path | None) -> dict[str, Any]:
    """Use installed manifest contracts when the development recipes are absent."""
    model_types = [config.get("model_type")]
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict) and text_cfg.get("model_type"):
        model_types.append(text_cfg["model_type"])
    seen_families = set()
    for path, manifest in manifests:
        family = manifest.get("family")
        check_types = manifest.get("hf_config_check", {}).get("model_type", [])
        if not family or not any(t in check_types for t in model_types if t):
            continue
        seen_families.add(family)
        diffs = _config_manifest_differences(config, manifest)
        result["selection"]["family_candidates"].append({
            "source": str(path), "bundle": path.parent.name,
            "family": family, "contract_matches": not diffs,
            "contract_differences": diffs,
            "runtime_manifest_ready": not diffs,
            "runtime_manifest_differences": diffs,
            "match_basis": "manifest model_type + hf_config_check (recipes unavailable)",
        })
    for family in sorted(seen_families):
        runtime_family = RUNTIME_FAMILY.get(family)
        donors = _registry_family_entries(model_list, runtime_family)
        for info in donors or []:
            name = info.get("name")
            if not name:
                continue
            for root in {Path(p).parent.parent for p, _ in manifests}:
                bundle = root / name
                if bundle.is_dir():
                    result["curated"]["family_xclbin_donors"].append({
                        "name": name, "path": str(bundle), "family": runtime_family,
                        "model_entry": name == result["identity"]["name"],
                    })
                    break
    if result["selection"]["family_candidates"]:
        result["requirements"]["kernel_family_candidates"] = sorted(seen_families)
        result["next_action"] = (
            "Installed manifests recognize this model_type. Recipe-level geometry and quant checks "
            "need an OpenFlowLM-Next checkout; this is not a model-list refusal, and config mismatches "
            "are diagnostics rather than model-name gates."
        )
    else:
        result["next_action"] = (
            "No installed manifest advertises this model_type. This is not a model-list refusal; "
            "use a development checkout for recipe detection or open a support issue with this report."
        )
    return result


def _registry_contains(path: Path | None, model_name: str) -> bool | None:
    """Whether a convenience model registry names this model; None if absent/bad."""
    if path is None or not path.is_file():
        return None
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
        models = registry.get("models", {})
        return any(info.get("name") == model_name
                   for sizes in models.values() if isinstance(sizes, dict)
                   for info in sizes.values() if isinstance(info, dict))
    except (OSError, ValueError, AttributeError):
        return None


def _registry_family_entries(path: Path | None, runtime_family: str | None) -> list[dict[str, Any]] | None:
    """Known-good convenience entries that can serve as family xclbin donors.

    The registry describes examples/install UX. Its lack of the candidate's
    model name is never a refusal; a family-level donor can still be used.
    """
    if path is None or not path.is_file() or not runtime_family:
        return None
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
        models = registry.get("models", {})
        return [info for sizes in models.values() if isinstance(sizes, dict)
                for info in sizes.values() if isinstance(info, dict)
                and info.get("details", {}).get("family") == runtime_family]
    except (OSError, ValueError, AttributeError):
        return None


def _default_model_list(recipe_root: Path | None) -> Path | None:
    candidates = []
    if recipe_root is not None:
        candidates.append(recipe_root.parent / "src" / "model_list.json")
    candidates.extend((Path("/opt/openflowlm/share/oflm/model_list.json"),
                       Path("/usr/share/oflm/model_list.json"),
                       Path("/usr/local/share/oflm/model_list.json")))
    return next((p for p in candidates if p.is_file()), None)


def inspect_model(model_dir: Path, roots: list[Path], recipes: Path | None = None,
                  model_list: Path | None = None) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    config, header, real_vocab = _load_model(model_dir)
    recipe_root = _recipe_root(str(recipes) if recipes else None)
    model_list = model_list or _default_model_list(recipe_root)
    manifests = _load_manifests(_manifest_paths(roots))

    result: dict[str, Any] = {
        "model": str(model_dir),
        "identity": {"name": model_dir.name, "model_type": config.get("model_type"),
                     "architectures": config.get("architectures", [])},
        "requirements": {"config_model_type": config.get("model_type"),
                          "padded_vocab": config.get("vocab_size"),
                          "real_vocab": real_vocab,
                          "tensor_count": sum(1 for k, v in header.items()
                                               if k != "__metadata__" and isinstance(v, dict))},
        "curated": {"model_list_entry": _registry_contains(model_list, model_dir.name),
                    "checked_in_spec": None,
                    "family_xclbin_donors": []},
        "selection": {"exact": [], "family_shape": [], "family_candidates": [], "rejected": []},
        "recipe": {"available": bool(recipe_root), "root": str(recipe_root) if recipe_root else None},
        "next_action": None,
    }

    if recipe_root is None:
        result["recipe"]["status"] = "unavailable"
        result["requirements"]["config_family"] = None
        return _inspect_without_recipes(result, config, manifests, model_list)

    try:
        spec, derived = _model_spec(model_dir, recipe_root)
    except Exception as exc:
        result["recipe"].update({"status": "refused", "error": f"{type(exc).__name__}: {exc}"})
        result["next_action"] = "Use this concrete recipe error to identify the unsupported architecture or geometry."
        return result

    candidate = derived["manifest"]
    model_type = str(spec.extra.get("model_type", "")).lower()
    runtime_family = MODEL_TYPE_RUNTIME_FAMILY.get(model_type) or RUNTIME_FAMILY.get(spec.family)
    result["requirements"].update({"kernel_family": spec.family,
                                   "runtime_family": runtime_family,
                                   "spec_hash": spec.spec_hash(),
                                   "quant": spec.canonical_quant(),
                                   "recipe_status": derived["recipe_status"]})
    result["recipe"].update({"status": derived["recipe_status"], "module": derived["recipe_module"]})

    specs_dir = recipe_root / "recipes" / "specs"
    for path in sorted(specs_dir.glob("*.json")):
        try:
            checked = json.loads(path.read_text(encoding="utf-8"))
            if checked.get("spec_hash") == spec.spec_hash():
                result["curated"]["checked_in_spec"] = path.name
                break
            from recipes.spec import ModelSpec
            if ModelSpec.from_dict(checked).spec_hash() == spec.spec_hash():
                result["curated"]["checked_in_spec"] = path.name
                break
        except Exception:
            continue

    # model_list is a donor directory catalogue, not a prerequisite for this
    # model. A compatible family entry can identify a shipped xclbin bundle
    # even when this model has never appeared in the registry.
    entries = _registry_family_entries(model_list, runtime_family)
    if entries is not None:
        for info in entries:
            name = info.get("name")
            if not isinstance(name, str) or not name:
                continue
            for root in roots:
                bundle = root / name
                if bundle.is_dir():
                    result["curated"]["family_xclbin_donors"].append({
                        "name": name, "path": str(bundle),
                        "family": runtime_family,
                        "model_entry": name == model_dir.name,
                    })
                    break

    for path, installed in manifests:
        source = path.parent.name
        installed_hash = installed.get("spec_hash")
        record = {"source": str(path), "bundle": source, "spec_hash": installed_hash}
        if installed_hash == spec.spec_hash():
            result["selection"]["exact"].append(record)
            result["selection"]["family_candidates"].append({
                **record, "contract_matches": True, "contract_differences": [],
                "runtime_manifest_differences": [],
            })
            continue
        if installed.get("family") != spec.family:
            result["selection"]["rejected"].append({**record, "reason": "different kernel family"})
            continue
        differences = _diff_paths(_kernel_contract(candidate), _kernel_contract(installed))
        runtime_differences = _diff_paths(candidate.get("layout", {}), installed.get("layout", {}))
        record["runtime_manifest_differences"] = runtime_differences
        record["runtime_manifest_ready"] = not runtime_differences
        record["contract_matches"] = not differences
        record["contract_differences"] = differences[:32]
        # Same-family bundles remain visible as candidates even when the
        # descriptors differ. These differences are evidence for the runtime
        # or user to evaluate, not a pre-approval wall. A compatible runtime
        # may consume a broader family xclbin than this exemplar manifest says.
        result["selection"]["family_candidates"].append(record)
        if not differences:
            result["selection"]["family_shape"].append(record)

    # Known-good is evidence, never eligibility. The recipe/catalogue verdict
    # remains independent of model_list membership or a checked-in spec.
    exact = result["selection"]["exact"]
    compat = result["selection"]["family_shape"]
    if exact:
        result["next_action"] = "Exact manifest match available; hash is provenance, not an eligibility rule."
    elif compat:
        result["next_action"] = "Family/shape kernel contract matches despite hash mismatch; inspect runtime_manifest_differences before linking."
    elif result["selection"]["family_candidates"]:
        result["next_action"] = "Same-family kernel bundles exist despite descriptor differences; they are candidates, not a refusal. Try the family bundle and let concrete runtime errors identify incompatibilities."
    elif result["curated"]["family_xclbin_donors"]:
        result["next_action"] = "No manifest-equivalent open set found; known-good family xclbins are available as a donor, regardless of this model's registry status."
    elif derived["recipe_status"] == "validated":
        result["next_action"] = "Recipe accepts this model; build/export a family variant or use an intentionally chosen xclbin source."
    elif derived["recipe_status"] == "catalogue-unvalidated":
        result["next_action"] = "Recipe is buildable only as an explicit unvalidated experiment; run its fixture before calling it supported."
    else:
        result["next_action"] = "Recipe refused this shape; report the named structural/catalogue constraint."
    return result


def _default_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("OFLM_XCLBIN_PATH")
    if env:
        roots.extend(Path(p) for p in env.split(os.pathsep) if p)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "xclbins"
        if candidate.is_dir():
            roots.append(candidate)
            break
    for p in (Path("/opt/openflowlm/share/oflm/xclbins"), Path("/usr/share/oflm/xclbins"),
              Path("/usr/local/share/oflm/xclbins")):
        if p.is_dir():
            roots.append(p)
    return roots


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect actual model requirements versus curated and compatible open kernels.")
    ap.add_argument("model_dir", type=Path, help="local q4nx model directory")
    ap.add_argument("--xclbin-root", action="append", type=Path, default=[],
                    help="xclbins root to scan; repeatable")
    ap.add_argument("--recipes", type=Path, help="open_kernels checkout (default: discover from env/repo/cwd)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ap.add_argument("--model-list", type=Path,
                    help="optional curated convenience registry to report separately from compatibility")
    ap.add_argument("--issue-base", help="issue creation URL; report only, never opens a browser")
    args = ap.parse_args(argv)
    try:
        report = inspect_model(args.model_dir, args.xclbin_root or _default_roots(),
                               args.recipes, args.model_list)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"oflm-support: {exc}", file=sys.stderr)
        return 2
    issue_base = args.issue_base or os.environ.get("OFLM_SUPPORT_ISSUE_URL")
    if issue_base and not report["selection"]["exact"] and not report["selection"]["family_candidates"] \
            and not report["curated"]["family_xclbin_donors"]:
        query = urllib.parse.urlencode({"title": f"Support {report['identity']['name']}",
                                        "body": json.dumps(report, indent=2)})
        report["issue_url"] = issue_base + ("&" if "?" in issue_base else "?") + query
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"Model: {report['identity']['name']}")
    print(f"Config model_type: {report['identity']['model_type'] or '(missing)'}")
    req = report["requirements"]
    print(f"Kernel family: {req.get('kernel_family', '(undetermined)')}")
    print(f"Recipe: {report['recipe'].get('status', '(unavailable)')}")
    print(f"Curated checked-in spec: {report['curated']['checked_in_spec'] or 'no'}")
    print(f"Exact manifests: {len(report['selection']['exact'])}")
    print(f"Family/shape candidates: {len(report['selection']['family_shape'])}")
    print(f"Same-family candidates: {len(report['selection']['family_candidates'])}")
    for match in report["selection"]["family_shape"]:
        print(f"  {match['bundle']}: runtime manifest {'ready' if match['runtime_manifest_ready'] else 'differs at ' + ', '.join(match['runtime_manifest_differences'][:6])}")
    print("Next:", report["next_action"])
    if report.get("issue_url"):
        print("Issue:", report["issue_url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Submit a local `oflm bench` CSV to the community leaderboard.

This is the CLI half of "Both, form first" (the web form is docs/benchmarks-submit.md).
It auto-fills everything that is knowable locally (CPU, OS, RAM, OFLM version)
and mints the same 10-min HMAC token the form uses, so it stays "trusted-key only"
without bundling a static secret in the repo. Usage:

    python3 utilities/bench-submit/submit.py bench_qwen3.6-35B-A3B_20261014.csv \
        --api https://oflm-bench.<you>.workers.dev --model Qwen3.6-35B-A3B \
        --hardware "Radeon 890M" --backend "NPU (XDNA2)" --quantization Q4_K_M

Then verify: curl "$API/api/benchmarks?model=Qwen3.6-35B-A3B&status=validated"
"""
from __future__ import annotations
import argparse, csv, json, os, platform, re, sys, urllib.request, uuid


def read_csv(path: str) -> str:
    with open(path, newline="", encoding="utf-8") as fh:
        return fh.read()


def local_cpu() -> str:
    try:
        if platform.system() == "Windows":
            try:
                import subprocess
                out = subprocess.check_output(["wmic", "cpu", "get", "name"], text=True, timeout=5)
                for line in out.splitlines():
                    if line.strip() and "Name" not in line:
                        return line.strip()
            except Exception:
                pass
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="ignore"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def local_ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            import subprocess
            out = subprocess.check_output(["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                                          text=True, timeout=5)
            for line in out.splitlines():
                if line.strip().isdigit():
                    return round(int(line.strip()) / (1024 ** 3), 1)
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        pass
    return None


def oflm_version() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["oflm", "--version"], text=True, timeout=5)
        m = re.search(r"(\d+\.\d+[\.\d]*)", out)
        return m.group(1) if m else "0.2.0"
    except Exception:
        return "0.2.0"


def read_consent() -> dict:
    """Read the anonymous install id oflm bench --submit would send.

    The file oflm writes is ~/.config/oflm/bench_consent.json on Linux,
    ~/.oflm/bench_consent.json on Windows (get_models_directory()).
    """
    import pathlib
    home = pathlib.Path.home()
    for base in [home / ".config" / "oflm", home / ".oflm"]:
        f = base / "bench_consent.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def api_post(url: str, payload: dict, token: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "X-OFLM-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get_token(api: str) -> str:
    with urllib.request.urlopen(f"{api.rstrip('/')}/api/token", timeout=15) as r:
        return json.loads(r.read())["token"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path")
    ap.add_argument("--api", required=True, help="Worker base URL, e.g. https://oflm-bench.x.workers.dev")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--task", default="LLM", choices=["LLM", "Embedding", "Vision-Language", "Audio", "ImageGen"])
    ap.add_argument("--backend", default="")
    ap.add_argument("--quantization", default="")
    ap.add_argument("--os", default=f"{platform.system()} {platform.release()}")
    ap.add_argument("--cpu", default=local_cpu())
    ap.add_argument("--ram-gb", type=float, default=local_ram_gb() or 0)
    ap.add_argument("--runs", type=int, default=0, help="measured iterations (from --bench-iterations)")
    ap.add_argument("--submission-id", default=str(uuid.uuid4()))
    args = ap.parse_args()
    consent = read_consent()
    payload = {"submission_id": args.submission_id, "model": args.model, "task": args.task,
               "install_id": consent.get("install_id"),
               "survey_opt_in": bool(consent.get("survey_opt_in")),
               "hardware": args.hardware, "cpu": args.cpu, "ram_gb": args.ram_gb or None,
               "backend": args.backend, "quantization": args.quantization,
               "oflm_version": oflm_version(), "os": args.os, "runs": args.runs,
               "csv": read_csv(args.csv_path)}
    token = get_token(args.api)
    status, body = api_post(f"{args.api.rstrip('/')}/api/benchmarks", payload, token)
    print(json.dumps({"http": status, **body}, indent=2))
    return 0 if status == 201 else 1


if __name__ == "__main__":
    sys.exit(main())

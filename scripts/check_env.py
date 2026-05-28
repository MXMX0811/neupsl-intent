#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Check Java and pslpython availability for mnist-add-minreal.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def check_environment() -> dict:
    java_path = shutil.which("java")
    java_ok = java_path is not None
    java_version = ""
    if java_ok:
        completed = subprocess.run(["java", "-version"], text=True, capture_output=True, check=False)
        java_version = (completed.stderr or completed.stdout).strip().splitlines()[0]

    pslpython_ok = False
    pslpython_runtime_ok = False
    pslpython_version = "unknown"
    pslpython_error = ""
    try:
        import pslpython

        pslpython_ok = True
        pslpython_version = getattr(pslpython, "__version__", "unknown")
        import pslpython.runtime  # noqa: F401

        pslpython_runtime_ok = True
    except Exception as exc:
        pslpython_error = f"{type(exc).__name__}: {exc}"

    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "java_ok": java_ok,
        "java_path": java_path,
        "java_version": java_version,
        "pslpython_ok": pslpython_ok,
        "pslpython_runtime_ok": pslpython_runtime_ok,
        "pslpython_version": pslpython_version,
        "pslpython_error": pslpython_error,
        "ready": java_ok and pslpython_ok and pslpython_runtime_ok,
    }


def main() -> None:
    args = parse_args()
    status = check_environment()
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return

    print(f"Python: {status['python']} ({status['python_version']})")
    print(f"Java: {'OK' if status['java_ok'] else 'MISSING'}")
    if status["java_ok"]:
        print(f"  {status['java_path']}")
        print(f"  {status['java_version']}")
    print(f"pslpython: {'OK' if status['pslpython_ok'] else 'MISSING'} ({status['pslpython_version']})")
    print(f"pslpython.runtime: {'OK' if status['pslpython_runtime_ok'] else 'MISSING'}")
    if status["pslpython_error"]:
        print(f"  {status['pslpython_error']}")
    print(f"Ready for NeuPSL runtime: {status['ready']}")

    if not status["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

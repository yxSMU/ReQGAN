#!/usr/bin/env python3
"""Run a named experiment with optional parameter overrides."""

from pathlib import Path
import json, runpy, sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "list"):
    print("Usage: python run.py EXPERIMENT_NAME [extra CLI arguments]")
    print("Experiments:")
    for name in sorted(manifest):
        print(" ", name, "->", manifest[name]["script"])
    sys.exit(0)
name = sys.argv[1]
if name not in manifest:
    raise SystemExit(f"Unknown experiment: {name}. Run python run.py list")
entry = manifest[name]
args = []
for key, value in entry["parameters"].items():
    if value is None or isinstance(value, bool):
        raise SystemExit(
            f"Preset {name} has a non-scalar parameter {key}={value!r}; inspect manifest"
        )
    args.extend((key, str(value)))
    # Explicit user arguments override any preset by keeping the last occurrence.
args.extend(sys.argv[2:])
sys.path.insert(0, str(root))
sys.argv = [str(root / entry["script"]), *args]
runpy.run_path(str(root / entry["script"]), run_name="__main__")

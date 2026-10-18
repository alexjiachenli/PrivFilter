#!/usr/bin/env python3
"""Check source integrity, portable paths and excluded runtime artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {
    "experiments",
    "slurm",
    "data",
    "outputs",
    "saved_models",
    "results",
    "noise_seeds",
    "shares",
    "Player-Data",
    "node_modules",
    "run_outputs",
}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".log",
    ".pem",
    ".key",
    ".pdf",
    ".png",
    ".jpg",
    ".zip",
    ".mp4",
}
ABANDONED_MODULES = {
    "stateful_detector",
    "ordinal_detector",
    "paired_bn_detector",
    "paired_bn_ordinal_delta_detector",
    "paired_bn_feature_bank",
    "head_pressure_feature_bank",
    "public_anchor_dual_view_detector",
    "patch_control_energy_detector",
    "multi_probe_patch_control_detector",
    "public_shared_bn_fixed_tau_detector",
    "paired_margin_detector",
    "detector_quick_fail",
    "mpc_consistency_screen",
}


def release_files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted({p.decode() for p in result.stdout.split(b"\0") if p})


def validate():
    errors = []
    paths = release_files()
    for name in paths:
        path = ROOT / name
        if name == "external/CryptGPU":
            continue
        if not path.is_file():
            errors.append(f"{name}: missing file or unsupported link")
            continue
        parts = Path(name).parts
        if set(parts) & FORBIDDEN_DIRS or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"{name}: generated/private/history artifact in release tree")
        if path.is_symlink():
            errors.append(f"{name}: symlinks are not part of this portable release")
        if path.stat().st_size > 1024 * 1024:
            errors.append(f"{name}: unexpected file larger than 1 MiB")
        if path.suffix not in {".py", ".yaml", ".yml", ".json", ".toml", ".txt", ".md", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8")
        for prefix in ("/" + "home/", "/" + "Users/", "C:" + "\\Users\\"):
            if prefix in text:
                errors.append(f"{name}: machine-specific absolute path")
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=name)
            except SyntaxError as exc:
                errors.append(f"{name}:{exc.lineno}: {exc.msg}")
                continue
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [v.name.split(".")[0] for v in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module.split(".")[0]]
                for module in modules:
                    if module in ABANDONED_MODULES:
                        errors.append(f"{name}:{node.lineno}: abandoned dependency {module}")
    manifest = ROOT / "SOURCE_MANIFEST.json"
    if not manifest.is_file():
        errors.append("SOURCE_MANIFEST.json: source provenance missing")
    else:
        data = json.loads(manifest.read_text())
        for item in data.get("files", []):
            source_file = ROOT / item["path"]
            if not source_file.is_file():
                errors.append(f"manifest references missing file: {item['path']}")
            elif hashlib.sha256(source_file.read_bytes()).hexdigest() != item["sha256"]:
                errors.append(f"source file changed since extraction snapshot: {item['path']}")
    gitlink = subprocess.run(
        ["git", "ls-files", "--stage", "external/CryptGPU"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pin = "2ff57b2b4d718f9665f4b2ac8245b0bcd7e65165"
    if not gitlink.startswith("160000 " + pin):
        errors.append("CryptGPU gitlink does not match the documented upstream pin")
    if errors:
        print("\n".join("FAIL " + e for e in errors))
        return 1
    print(
        f"PASS: {len(paths)} release entries; syntax, portability, artifact exclusions, provenance, CryptGPU pin"
    )
    return 0


if __name__ == "__main__":
    sys.exit(validate())

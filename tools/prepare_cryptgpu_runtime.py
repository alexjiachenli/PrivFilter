#!/usr/bin/env python3
"""Prepare a PyTorch-2-compatible runtime from the pinned CryptGPU source.

Export the gitlink commit, apply the compatibility patch and record source,
patch and resulting-tree hashes in .cache.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cryptgpu_runtime as runtime


def run(
    arguments: list[str],
    *,
    cwd: Path,
    input_bytes: bytes | None = None,
    env: dict[str, str] | None = None,
) -> bytes:
    try:
        result = subprocess.run(
            arguments,
            cwd=str(cwd),
            input=input_bytes,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", b"") or b""
        stdout = getattr(error, "stdout", b"") or b""
        detail = (
            stderr.decode("utf-8", errors="replace").strip()
            or stdout.decode("utf-8", errors="replace").strip()
            or str(error)
        )
        raise runtime.CryptGPURuntimeError(
            f"Command failed ({shlex.join(arguments)}): {detail}"
        ) from error
    return result.stdout


def patched_files(patch_path: Path) -> list[str]:
    pattern = re.compile(r"^\+\+\+ b/(.+)$", flags=re.MULTILINE)
    return sorted(set(pattern.findall(patch_path.read_text(encoding="utf-8"))))


def _safe_extract(archive_bytes: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        try:
            archive.extractall(destination, filter="data")
        except TypeError:  # Python < 3.12
            archive.extractall(destination)


def _metadata(
    *,
    identity: dict[str, str],
    archive_sha256: str,
    runtime_root: Path,
    patch_path: Path,
) -> dict:
    return {
        "schema": runtime.METADATA_SCHEMA,
        "source": {
            "gitlink_path": identity["gitlink_path"],
            "commit": identity["commit"],
            "tree": identity["tree"],
            "archive_sha256": archive_sha256,
        },
        "patch": {
            "path": identity["patch_path"],
            "sha256": identity["patch_sha256"],
            "files": patched_files(patch_path),
            "scope": [
                "optional torchcsprng import with fail-closed explicit use",
                "newer-PyTorch serialization source lookup fallback",
                "PyTorch-2 ONNX export and symbolic registry compatibility",
                "ONNX Identity conversion",
                "bounded multiprocess completion with complete rank packets",
                "remove the unused plaintext opening from encrypted exponential",
                "remove unused plaintext openings from encrypted maximum",
            ],
            "excluded": [
                "device-selection changes",
                "debug printing",
                "GPU synchronization instrumentation",
                "detector decision-rule or secure-arithmetic changes",
            ],
        },
        "runtime": {
            "cache_key": runtime.cache_key(identity),
            "tree_sha256": runtime.tree_sha256(runtime_root),
        },
    }


def prepare(
    *,
    repo_root: Path = REPO_ROOT,
    patch_path: Path = runtime.PATCH_PATH,
    cache_root: Path | None = None,
    force: bool = False,
) -> Path:
    repo_root = Path(repo_root).resolve()
    patch_path = Path(patch_path).resolve()
    cache_base = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else repo_root / ".cache" / "cryptgpu-runtime"
    )
    identity = runtime.source_identity(repo_root=repo_root, patch_path=patch_path)
    target_parent = cache_base / runtime.cache_key(identity)
    target = target_parent / "CryptGPU"
    cache_base.mkdir(parents=True, exist_ok=True)

    lock_path = cache_base / ".prepare.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists() and not force:
            return runtime.validate_runtime_root(
                target,
                expected_identity=identity,
                require_metadata=True,
                verify_tree=True,
            )

        external_root = runtime.source_repository(
            repo_root=repo_root,
            commit=identity["commit"],
        )
        archive = run(
            ["git", "archive", "--format=tar", identity["commit"]],
            cwd=external_root,
        )
        archive_sha256 = hashlib.sha256(archive).hexdigest()

        with tempfile.TemporaryDirectory(
            prefix=f".{runtime.cache_key(identity)}-", dir=cache_base
        ) as temporary:
            temporary_path = Path(temporary)
            build_root = temporary_path / "CryptGPU"
            build_root.mkdir()
            _safe_extract(archive, build_root)
            # Apply the compatibility patch with Git.
            patch_env = os.environ.copy()
            # Restrict Git discovery to the exported runtime directory.
            patch_env["GIT_CEILING_DIRECTORIES"] = str(build_root.parent)
            run(["git", "apply", "--check", str(patch_path)], cwd=build_root, env=patch_env)
            run(["git", "apply", str(patch_path)], cwd=build_root, env=patch_env)
            run(
                ["git", "apply", "--reverse", "--check", str(patch_path)],
                cwd=build_root,
                env=patch_env,
            )
            metadata = _metadata(
                identity=identity,
                archive_sha256=archive_sha256,
                runtime_root=build_root,
                patch_path=patch_path,
            )
            (build_root / runtime.METADATA_NAME).write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            runtime.validate_runtime_root(
                build_root,
                expected_identity=identity,
                require_metadata=True,
                verify_tree=True,
            )

            if target_parent.exists():
                shutil.rmtree(target_parent)
            target_parent.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_path, target_parent)

        return runtime.validate_runtime_root(
            target,
            expected_identity=identity,
            require_metadata=True,
            verify_tree=True,
        )


def verify_import(runtime_root: Path, *, verify_conversion: bool = False) -> None:
    statements = [
        "import crypten",
        "print(crypten.__version__)",
    ]
    if verify_conversion:
        statements.extend(
            [
                "import torch",
                "model=torch.nn.Sequential(torch.nn.Identity(),torch.nn.Linear(4,2)).eval()",
                "converted=crypten.nn.from_pytorch(model,torch.zeros(1,4))",
                "assert converted is not None",
            ]
        )
    environment = os.environ.copy()
    environment[runtime.ENVIRONMENT_VARIABLE] = str(runtime_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(runtime_root), str(REPO_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    run([sys.executable, "-c", ";".join(statements)], cwd=REPO_ROOT, env=environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-import-check", action="store_true")
    parser.add_argument("--verify-conversion", action="store_true")
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--shell",
        action="store_true",
        help="print a shell export for the prepared runtime",
    )
    output.add_argument("--json", action="store_true", help="print metadata JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = prepare(cache_root=args.cache_root, force=args.force)
    if not args.skip_import_check:
        verify_import(root, verify_conversion=args.verify_conversion)
    if args.shell:
        print(f"export {runtime.ENVIRONMENT_VARIABLE}={shlex.quote(str(root))}")
    elif args.json:
        print(json.dumps(runtime.load_metadata(root), indent=2, sort_keys=True))
    else:
        print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

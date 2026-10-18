"""Select and activate the pinned CryptGPU runtime.

The preparer exports external/CryptGPU plus the compatibility patch to .cache.
FLDD_CRYPTGPU_ROOT selects a separately prepared installation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, MutableMapping, Optional


REPO_ROOT = Path(__file__).resolve().parent
GITLINK_PATH = Path("external/CryptGPU")
PATCH_PATH = REPO_ROOT / "patches" / "cryptgpu-pytorch2-runtime.patch"
DEFAULT_CACHE_ROOT = REPO_ROOT / ".cache" / "cryptgpu-runtime"
ENVIRONMENT_VARIABLE = "FLDD_CRYPTGPU_ROOT"
SOURCE_REPOSITORY_ENVIRONMENT_VARIABLE = "FLDD_CRYPTGPU_SOURCE_REPOSITORY"
METADATA_NAME = ".fldd-runtime.json"
METADATA_SCHEMA = "fldd-cryptgpu-runtime-v1"


class CryptGPURuntimeError(RuntimeError):
    """CryptGPU runtime selection or integrity-check failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path, *, exclude_metadata: bool = True) -> str:
    """Hash relative paths, entry types, and contents in deterministic order."""

    root = Path(root).resolve()
    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for entry in entries:
        relative = entry.relative_to(root).as_posix()
        if exclude_metadata and relative == METADATA_NAME:
            continue
        # Exclude interpreter bytecode from the source-tree digest.
        if "__pycache__" in entry.relative_to(root).parts or entry.suffix in {
            ".pyc",
            ".pyo",
        }:
            continue
        if entry.is_symlink():
            kind = b"L"
            payload = os.readlink(entry).encode("utf-8", errors="surrogateescape")
        elif entry.is_file():
            kind = b"F"
            payload = entry.read_bytes()
        elif entry.is_dir():
            kind = b"D"
            payload = b""
        else:
            raise CryptGPURuntimeError(f"Unsupported runtime entry: {entry}")
        digest.update(kind)
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_output(arguments: list[str], *, cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise CryptGPURuntimeError(f"Git identity lookup failed: {detail.strip()}") from error
    return result.stdout.strip()


def source_repository(
    *,
    repo_root: Path = REPO_ROOT,
    commit: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    """Locate an existing Git object store containing the pinned CryptGPU commit.

    Search the explicit source checkout, current submodule and sibling worktrees.
    The preparer exports the pinned Git object from the selected store.
    """

    repo_root = Path(repo_root).resolve()
    environment = os.environ if environ is None else environ
    pinned_commit = commit or _git_output(
        ["rev-parse", f"HEAD:{GITLINK_PATH.as_posix()}"], cwd=repo_root
    )
    explicit = str(environment.get(SOURCE_REPOSITORY_ENVIRONMENT_VARIABLE, "")).strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    else:
        candidates.append(repo_root / GITLINK_PATH)
        try:
            worktree_listing = _git_output(["worktree", "list", "--porcelain"], cwd=repo_root)
        except CryptGPURuntimeError:
            worktree_listing = ""
        for line in worktree_listing.splitlines():
            if line.startswith("worktree "):
                candidates.append(Path(line[len("worktree ") :]) / GITLINK_PATH)

    seen: set[Path] = set()
    failures: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_dir():
            failures.append(f"{resolved}: not a directory")
            continue
        try:
            top_level = Path(_git_output(["rev-parse", "--show-toplevel"], cwd=resolved)).resolve()
            if top_level != resolved:
                failures.append(f"{resolved}: not an independent Git checkout")
                continue
            object_type = _git_output(["cat-file", "-t", pinned_commit], cwd=resolved)
            if object_type != "commit":
                failures.append(f"{resolved}: pinned object is {object_type!r}")
                continue
        except CryptGPURuntimeError as error:
            failures.append(f"{resolved}: {error}")
            continue
        return resolved

    detail = "; ".join(failures) if failures else "no candidate checkout found"
    raise CryptGPURuntimeError(
        "Could not locate a CryptGPU Git object store for pinned commit "
        f"{pinned_commit}. Set {SOURCE_REPOSITORY_ENVIRONMENT_VARIABLE} to an "
        f"independent CryptGPU checkout containing that object. Candidates: {detail}"
    )


def source_identity(
    *, repo_root: Path = REPO_ROOT, patch_path: Path = PATCH_PATH
) -> dict[str, str]:
    """Return the gitlink commit, source tree and patch hashes."""

    repo_root = Path(repo_root).resolve()
    patch_path = Path(patch_path).resolve()
    commit = _git_output(["rev-parse", f"HEAD:{GITLINK_PATH.as_posix()}"], cwd=repo_root)
    external_root = source_repository(repo_root=repo_root, commit=commit)
    object_type = _git_output(["cat-file", "-t", commit], cwd=external_root)
    if object_type != "commit":
        raise CryptGPURuntimeError(
            f"Root gitlink {GITLINK_PATH} resolved to {object_type}, not a commit"
        )
    tree = _git_output(["rev-parse", f"{commit}^{{tree}}"], cwd=external_root)
    return {
        "gitlink_path": GITLINK_PATH.as_posix(),
        "commit": commit,
        "tree": tree,
        "patch_path": patch_path.relative_to(repo_root).as_posix(),
        "patch_sha256": sha256_file(patch_path),
    }


def cache_key(identity: Mapping[str, str]) -> str:
    return f"{identity['commit'][:12]}-{identity['patch_sha256'][:12]}"


def default_runtime_root(
    *,
    repo_root: Path = REPO_ROOT,
    patch_path: Path = PATCH_PATH,
    cache_root: Optional[Path] = None,
) -> Path:
    identity = source_identity(repo_root=repo_root, patch_path=patch_path)
    base = (
        Path(cache_root).resolve()
        if cache_root
        else Path(repo_root).resolve() / ".cache" / "cryptgpu-runtime"
    )
    return base / cache_key(identity) / "CryptGPU"


def load_metadata(runtime_root: Path) -> dict:
    metadata_path = Path(runtime_root) / METADATA_NAME
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CryptGPURuntimeError(
            f"Missing or invalid CryptGPU runtime metadata: {metadata_path}"
        ) from error
    if not isinstance(value, dict):
        raise CryptGPURuntimeError(f"CryptGPU metadata is not an object: {metadata_path}")
    return value


def validate_runtime_root(
    runtime_root: Path,
    *,
    expected_identity: Optional[Mapping[str, str]] = None,
    require_metadata: bool = True,
    verify_tree: bool = False,
) -> Path:
    """Validate the package layout and prepared-runtime hashes."""

    runtime_root = Path(runtime_root).expanduser().resolve()
    package_marker = runtime_root / "crypten" / "__init__.py"
    if not package_marker.is_file():
        raise CryptGPURuntimeError(
            f"Selected CryptGPU root does not contain crypten/__init__.py: {runtime_root}"
        )
    metadata_path = runtime_root / METADATA_NAME
    if not metadata_path.exists() and not require_metadata:
        return runtime_root

    metadata = load_metadata(runtime_root)
    if metadata.get("schema") != METADATA_SCHEMA:
        raise CryptGPURuntimeError(
            f"Unexpected CryptGPU metadata schema: {metadata.get('schema')!r}"
        )
    if expected_identity is not None:
        source = metadata.get("source", {})
        patch = metadata.get("patch", {})
        checks = {
            "source commit": (source.get("commit"), expected_identity["commit"]),
            "source tree": (source.get("tree"), expected_identity["tree"]),
            "patch SHA-256": (
                patch.get("sha256"),
                expected_identity["patch_sha256"],
            ),
        }
        mismatches = [
            f"{label}: recorded={actual!r}, expected={expected!r}"
            for label, (actual, expected) in checks.items()
            if actual != expected
        ]
        if mismatches:
            raise CryptGPURuntimeError(
                "CryptGPU runtime identity mismatch: " + "; ".join(mismatches)
            )
    if verify_tree:
        recorded = metadata.get("runtime", {}).get("tree_sha256")
        actual = tree_sha256(runtime_root)
        if not recorded or actual != recorded:
            raise CryptGPURuntimeError(
                f"CryptGPU runtime tree mismatch: recorded={recorded!r}, actual={actual!r}"
            )
    return runtime_root


def select_runtime_root(
    *,
    environ: Optional[Mapping[str, str]] = None,
    repo_root: Path = REPO_ROOT,
    patch_path: Path = PATCH_PATH,
    cache_root: Optional[Path] = None,
    verify_tree: bool = False,
) -> Path:
    """Select an explicit override or the deterministic prepared cache."""

    environment = os.environ if environ is None else environ
    override = environment.get(ENVIRONMENT_VARIABLE)
    if override:
        # Validate optional metadata for separately managed runtime installations.
        root = Path(override)
        has_metadata = (root.expanduser() / METADATA_NAME).exists()
        return validate_runtime_root(
            root, require_metadata=has_metadata, verify_tree=verify_tree and has_metadata
        )

    identity = source_identity(repo_root=repo_root, patch_path=patch_path)
    base = (
        Path(cache_root).resolve()
        if cache_root
        else Path(repo_root).resolve() / ".cache" / "cryptgpu-runtime"
    )
    root = base / cache_key(identity) / "CryptGPU"
    if not root.exists():
        raise CryptGPURuntimeError(
            "Prepared CryptGPU runtime is missing. Run "
            "`python tools/prepare_cryptgpu_runtime.py` first."
        )
    return validate_runtime_root(
        root,
        expected_identity=identity,
        require_metadata=True,
        verify_tree=verify_tree,
    )


def activate_runtime(
    *,
    environ: Optional[MutableMapping[str, str]] = None,
    path_list: Optional[list[str]] = None,
    verify_tree: bool = False,
) -> Path:
    """Select the runtime, export it to child processes, and prepend sys.path."""

    environment = os.environ if environ is None else environ
    paths = sys.path if path_list is None else path_list
    root = select_runtime_root(environ=environment, verify_tree=verify_tree)

    loaded = sys.modules.get("crypten")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
        try:
            loaded_file.relative_to(root)
        except ValueError as error:
            raise CryptGPURuntimeError(
                f"crypten is already imported from a different runtime: {loaded_file}"
            ) from error

    # Match MKL threading to libgomp in spawned workers; preserve explicit overrides.
    environment.setdefault("MKL_THREADING_LAYER", "GNU")

    root_string = str(root)
    paths[:] = [entry for entry in paths if entry != root_string]
    paths.insert(0, root_string)
    environment[ENVIRONMENT_VARIABLE] = root_string
    return root


__all__ = [
    "CryptGPURuntimeError",
    "ENVIRONMENT_VARIABLE",
    "METADATA_NAME",
    "METADATA_SCHEMA",
    "SOURCE_REPOSITORY_ENVIRONMENT_VARIABLE",
    "activate_runtime",
    "cache_key",
    "default_runtime_root",
    "load_metadata",
    "select_runtime_root",
    "source_identity",
    "source_repository",
    "tree_sha256",
    "validate_runtime_root",
]

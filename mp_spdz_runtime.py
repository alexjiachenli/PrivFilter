"""Coordinate MP-SPDZ workspaces and party execution.

A resolved runtime directory is locked across each input, compile and
execute transaction because programs and party inputs share that directory.
"""

from __future__ import annotations

import contextlib
import codecs
import fcntl
import hashlib
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple


LOCK_FILE_NAME = ".fldd-mp-spdz-runtime.lock"
RECEIPT_SCHEMA = "fldd-mp-spdz-execution-v1"
LOCK_RECEIPT_SCHEMA = "fldd-mp-spdz-runtime-lock-v2"

_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: Dict[str, threading.RLock] = {}
_THREAD_STATE = threading.local()
_ACTIVE_FDS_GUARD = threading.Lock()
_ACTIVE_LOCK_FDS: set[int] = set()


def _before_fork() -> None:
    """Freeze lock registries so the child observes one consistent snapshot."""

    _LOCKS_GUARD.acquire()
    _ACTIVE_FDS_GUARD.acquire()


def _after_fork_parent() -> None:
    _ACTIVE_FDS_GUARD.release()
    _LOCKS_GUARD.release()


def _after_fork_child() -> None:
    """Drop inherited flock descriptions and all process/thread lock state."""

    global _LOCKS_GUARD, _PROCESS_LOCKS, _THREAD_STATE
    global _ACTIVE_FDS_GUARD, _ACTIVE_LOCK_FDS

    # Close inherited flock descriptors so the child obtains its own lock.
    for fd in tuple(_ACTIVE_LOCK_FDS):
        try:
            os.close(fd)
        except OSError:
            pass
    _ACTIVE_LOCK_FDS = set()
    _PROCESS_LOCKS = {}
    _THREAD_STATE = threading.local()
    # Reinitialize thread locks after fork.
    _LOCKS_GUARD = threading.Lock()
    _ACTIVE_FDS_GUARD = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_parent,
        after_in_child=_after_fork_child,
    )


def canonical_mp_spdz_path(mp_spdz_path: str) -> str:
    """Return the physical backend path so symlinked worktrees share a lock."""

    return str(Path(mp_spdz_path).expanduser().resolve())


def runtime_lock_path(mp_spdz_path: str) -> str:
    return os.path.join(canonical_mp_spdz_path(mp_spdz_path), LOCK_FILE_NAME)


def _process_lock(canonical_path: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(canonical_path, threading.RLock())


def _held_depths() -> Dict[str, int]:
    depths = getattr(_THREAD_STATE, "held_depths", None)
    if depths is None:
        depths = {}
        _THREAD_STATE.held_depths = depths
    return depths


def mp_spdz_runtime_lock_is_held(mp_spdz_path: str) -> bool:
    return _held_depths().get(canonical_mp_spdz_path(mp_spdz_path), 0) > 0


def require_mp_spdz_runtime_lock(mp_spdz_path: str) -> None:
    if not mp_spdz_runtime_lock_is_held(mp_spdz_path):
        raise RuntimeError(
            "MP-SPDZ runtime transaction is not protected by the shared runtime lock"
        )


@contextlib.contextmanager
def mp_spdz_runtime_lock(
    mp_spdz_path: str,
    *,
    operation: str,
) -> Iterator[Dict[str, Any]]:
    """Hold a blocking cross-process lock for one complete MP-SPDZ transaction.

    A process-local RLock serializes threads and supports same-thread re-entry.
    """

    canonical_path = canonical_mp_spdz_path(mp_spdz_path)
    if not os.path.isdir(canonical_path):
        raise FileNotFoundError(f"MP-SPDZ path not found: {canonical_path}")
    lock_path = os.path.join(canonical_path, LOCK_FILE_NAME)
    local_lock = _process_lock(canonical_path)
    wait_started = time.monotonic()

    with local_lock:
        depths = _held_depths()
        if depths.get(canonical_path, 0) > 0:
            depths[canonical_path] += 1
            receipt = {
                "schema": LOCK_RECEIPT_SCHEMA,
                "backend_path": canonical_path,
                "lock_path": lock_path,
                "operation": str(operation),
                "wait_seconds": float(time.monotonic() - wait_started),
                "held_seconds": 0.0,
                "acquired": False,
                "released": False,
                "scope_exited": False,
                "outermost": False,
                "reentrant": True,
            }
            held_started = time.monotonic()
            try:
                yield receipt
            finally:
                receipt["held_seconds"] = float(time.monotonic() - held_started)
                receipt["scope_exited"] = True
                depths[canonical_path] -= 1
            return

        # Keep descriptor registration atomic with the at-fork snapshot.
        with _ACTIVE_FDS_GUARD:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            _ACTIVE_LOCK_FDS.add(fd)
        owner_pid = os.getpid()
        acquired = False
        receipt = {
            "schema": LOCK_RECEIPT_SCHEMA,
            "backend_path": canonical_path,
            "lock_path": lock_path,
            "operation": str(operation),
            "wait_seconds": 0.0,
            "held_seconds": 0.0,
            "acquired": False,
            "released": False,
            "scope_exited": False,
            "outermost": True,
            "reentrant": False,
        }
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            acquired = True
            receipt["wait_seconds"] = float(time.monotonic() - wait_started)
            receipt["acquired"] = True
            depths[canonical_path] = 1
            held_started = time.monotonic()
            try:
                yield receipt
            finally:
                receipt["held_seconds"] = float(time.monotonic() - held_started)
                depths.pop(canonical_path, None)
        finally:
            # An inherited context must not close a descriptor reused by the child.
            if os.getpid() == owner_pid:
                unlock_succeeded = False
                if acquired:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                        unlock_succeeded = True
                    except OSError:
                        pass
                close_succeeded = False
                with _ACTIVE_FDS_GUARD:
                    _ACTIVE_LOCK_FDS.discard(fd)
                    try:
                        os.close(fd)
                        close_succeeded = True
                    finally:
                        receipt["released"] = bool(
                            acquired and (unlock_succeeded or close_succeeded)
                        )
                        receipt["scope_exited"] = True


class MpSpdzPartyExecutionError(RuntimeError):
    """Party execution failure with a structured execution receipt."""

    def __init__(self, message: str, receipt: Dict[str, Any]):
        super().__init__(message)
        self.receipt = receipt


_DETERMINISTIC_PATTERNS = (
    re.compile(r"not enough inputs", re.IGNORECASE),
    re.compile(r"\bINPUTMIXED\b", re.IGNORECASE),
    re.compile(r"invalid (?:integer|number|numeric|input)", re.IGNORECASE),
    re.compile(r"\b(?:nan|inf|infinity)\b", re.IGNORECASE),
    re.compile(r"non[- ]finite", re.IGNORECASE),
    re.compile(r"overflow", re.IGNORECASE),
    re.compile(r"division by zero", re.IGNORECASE),
)
_TRANSIENT_PATTERNS = (
    re.compile(r"stream truncated", re.IGNORECASE),
    re.compile(r"connection (?:reset|refused|aborted)", re.IGNORECASE),
    re.compile(r"broken pipe", re.IGNORECASE),
    re.compile(r"address already in use", re.IGNORECASE),
    re.compile(r"network is unreachable", re.IGNORECASE),
    re.compile(r"transport endpoint", re.IGNORECASE),
    re.compile(r"TLS.*(?:read|write|handshake)", re.IGNORECASE),
    re.compile(r"SSL.*(?:read|write|handshake)", re.IGNORECASE),
)


def classify_party_failure(
    stderr_texts: Sequence[str],
    *,
    timed_out: bool,
    deterministic_marker: bool = False,
    transient_marker: bool = False,
) -> Tuple[str, bool]:
    """Classify only explicit communication/time failures as retryable."""

    combined = "\n".join(stderr_texts)
    if deterministic_marker or any(pattern.search(combined) for pattern in _DETERMINISTIC_PATTERNS):
        return "deterministic_input_or_numeric", False
    if timed_out:
        return "transient_timeout", True
    if transient_marker or any(pattern.search(combined) for pattern in _TRANSIENT_PATTERNS):
        return "transient_communication", True
    return "process_exit_unknown", False


def _bounded_file_metadata(handle: Any, max_excerpt_chars: int) -> Tuple[str, Dict[str, Any]]:
    handle.flush()
    handle.seek(0, os.SEEK_END)
    size = int(handle.tell())
    digest = hashlib.sha256()
    deterministic_marker = False
    transient_marker = False
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    scan_tail = ""
    handle.seek(0)
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        scan_text = scan_tail + decoder.decode(chunk, final=False)
        deterministic_marker = deterministic_marker or any(
            pattern.search(scan_text) for pattern in _DETERMINISTIC_PATTERNS
        )
        transient_marker = transient_marker or any(
            pattern.search(scan_text) for pattern in _TRANSIENT_PATTERNS
        )
        # Retain overlap for markers split across diagnostic chunks.
        scan_tail = scan_text[-4096:]
    final_text = scan_tail + decoder.decode(b"", final=True)
    deterministic_marker = deterministic_marker or any(
        pattern.search(final_text) for pattern in _DETERMINISTIC_PATTERNS
    )
    transient_marker = transient_marker or any(
        pattern.search(final_text) for pattern in _TRANSIENT_PATTERNS
    )

    half = max(1, max_excerpt_chars // 2)
    handle.seek(0)
    head = handle.read(half)
    if size <= max_excerpt_chars:
        handle.seek(0)
        excerpt_bytes = handle.read()
        truncated = False
    else:
        handle.seek(max(0, size - half))
        tail = handle.read(half)
        excerpt_bytes = head + b"\n...[stderr truncated]...\n" + tail
        truncated = True
    excerpt = excerpt_bytes.decode("utf-8", errors="replace")
    return excerpt, {
        "sha256": digest.hexdigest(),
        "byte_count": size,
        "truncated": truncated,
        "excerpt": excerpt,
        "failure_markers": [
            marker
            for marker, present in (
                ("deterministic_input_or_numeric", deterministic_marker),
                ("transient_communication", transient_marker),
            )
            if present
        ],
    }


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(int(process_group_id), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Treat inaccessible process groups as active.
        return True
    except OSError:
        return True


def _signal_process_groups(process_group_ids: Sequence[int], sig: int) -> None:
    for process_group_id in sorted(set(int(value) for value in process_group_ids)):
        try:
            os.killpg(process_group_id, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _wait_for_processes(processes: Sequence[subprocess.Popen], deadline: float) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass


def _terminate_and_reap(
    processes: Sequence[subprocess.Popen],
    process_group_ids: Sequence[int] | None = None,
    grace_seconds: float = 2.0,
) -> bool:
    """Terminate complete party process groups and reap every direct leader."""

    if process_group_ids is None:
        process_group_ids = [process.pid for process in processes]
    group_ids = sorted(set(int(value) for value in process_group_ids))
    _signal_process_groups(group_ids, signal.SIGTERM)
    term_deadline = time.monotonic() + max(0.0, float(grace_seconds))
    _wait_for_processes(processes, term_deadline)
    while time.monotonic() < term_deadline:
        if not any(_process_group_exists(group_id) for group_id in group_ids):
            break
        time.sleep(0.02)

    remaining_groups = [group_id for group_id in group_ids if _process_group_exists(group_id)]
    if remaining_groups:
        _signal_process_groups(remaining_groups, signal.SIGKILL)

    reap_deadline = time.monotonic() + 5.0
    _wait_for_processes(processes, reap_deadline)
    while time.monotonic() < reap_deadline:
        direct_reaped = all(process.poll() is not None for process in processes)
        groups_absent = not any(_process_group_exists(group_id) for group_id in group_ids)
        if direct_reaped and groups_absent:
            return True
        time.sleep(0.02)
    return all(process.poll() is not None for process in processes) and not any(
        _process_group_exists(group_id) for group_id in group_ids
    )


def wait_for_tcp_port_release(host: str, port: int, timeout_seconds: float = 5.0) -> bool:
    """Return only after the launcher port is bindable or the deadline expires."""

    return wait_for_tcp_ports_release(host, [port], timeout_seconds=timeout_seconds)


def wait_for_tcp_ports_release(
    host: str,
    ports: Sequence[int],
    timeout_seconds: float = 5.0,
) -> bool:
    """Return only when every requested TCP port can be bound simultaneously."""

    checked_ports = [int(port) for port in ports]
    if not checked_ports or any(port < 1 or port > 65535 for port in checked_ports):
        raise ValueError("TCP ports must be a non-empty sequence within 1..65535")

    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        sockets: List[socket.socket] = []
        try:
            for port in checked_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(sock)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
        finally:
            for sock in sockets:
                sock.close()
        time.sleep(0.05)


def _run_party_attempt(
    commands: Sequence[Sequence[str]],
    *,
    cwd: str,
    timeout_seconds: float,
    startup_delay_seconds: float,
    failure_grace_seconds: float,
    max_stderr_excerpt_chars: int,
) -> Tuple[str, Dict[str, Any]]:
    processes: List[subprocess.Popen] = []
    process_group_ids: List[int] = []
    stderr_files = [tempfile.TemporaryFile(mode="w+b") for _ in commands]
    p0_stdout_file = tempfile.TemporaryFile(mode="w+b")
    timed_out = False
    orphan_free = False
    launch_error = None
    started = time.monotonic()
    try:
        try:
            for party_id, command in enumerate(commands):
                process = subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    stdout=p0_stdout_file if party_id == 0 else subprocess.DEVNULL,
                    stderr=stderr_files[party_id],
                    start_new_session=True,
                )
                processes.append(process)
                # Save the process-group ID so descendants remain reachable after the leader exits.
                process_group_ids.append(process.pid)
                if party_id + 1 < len(commands):
                    time.sleep(max(0.0, float(startup_delay_seconds)))
        except Exception as exc:
            launch_error = f"{type(exc).__name__}: {exc}"

        deadline = started + max(0.0, float(timeout_seconds))
        first_failure_at = None
        while launch_error is None:
            returncodes = [process.poll() for process in processes]
            if len(processes) == len(commands) and all(code is not None for code in returncodes):
                break
            now = time.monotonic()
            if any(code not in (None, 0) for code in returncodes) and first_failure_at is None:
                first_failure_at = now
            if first_failure_at is not None and now - first_failure_at >= failure_grace_seconds:
                break
            if now >= deadline:
                timed_out = True
                break
            time.sleep(0.05)

        orphan_free = _terminate_and_reap(processes, process_group_ids)
        returncodes = [process.returncode for process in processes]
        returncodes.extend([None] * (len(commands) - len(returncodes)))

        p0_stdout_file.flush()
        p0_stdout_file.seek(0)
        p0_output = p0_stdout_file.read().decode("utf-8", errors="replace")
        stderr_texts: List[str] = []
        stderr_receipts = []
        deterministic_marker = False
        transient_marker = False
        for party_id, stderr_file in enumerate(stderr_files):
            excerpt, metadata = _bounded_file_metadata(stderr_file, max_stderr_excerpt_chars)
            stderr_texts.append(excerpt)
            marker_classes = metadata.get("failure_markers", [])
            deterministic_marker = deterministic_marker or (
                "deterministic_input_or_numeric" in marker_classes
            )
            transient_marker = transient_marker or ("transient_communication" in marker_classes)
            stderr_receipts.append({"party": party_id, **metadata})

        success = (
            launch_error is None
            and not timed_out
            and len(processes) == len(commands)
            and all(code == 0 for code in returncodes)
            and orphan_free
        )
        if success:
            failure_class, retryable = "none", False
        elif launch_error is not None:
            failure_class, retryable = "process_launch", False
        else:
            failure_class, retryable = classify_party_failure(
                stderr_texts,
                timed_out=timed_out,
                deterministic_marker=deterministic_marker,
                transient_marker=transient_marker,
            )
        receipt = {
            "status": "success" if success else "failed",
            "failure_class": failure_class,
            "retryable": bool(retryable),
            "timed_out": bool(timed_out),
            "launch_error": launch_error,
            "party_returncodes": returncodes,
            "party_stderr": stderr_receipts,
            "cleanup_complete": bool(orphan_free),
            "runtime_seconds": float(time.monotonic() - started),
        }
        return p0_output, receipt
    finally:
        if not orphan_free:
            _terminate_and_reap(processes, process_group_ids)
        p0_stdout_file.close()
        for stderr_file in stderr_files:
            stderr_file.close()


def run_mp_spdz_parties(
    commands: Sequence[Sequence[str]],
    *,
    mp_spdz_path: str,
    base_port: int,
    timeout_seconds: float,
    max_transient_retries: int = 0,
    retry_backoff_seconds: float = 1.0,
    startup_delay_seconds: float = 0.1,
    failure_grace_seconds: float = 1.0,
    max_stderr_excerpt_chars: int = 8192,
) -> Tuple[str, Dict[str, Any]]:
    """Run all parties, capture bounded diagnostics, and retry explicit transients."""

    require_mp_spdz_runtime_lock(mp_spdz_path)
    if len(commands) != 3:
        raise ValueError("FLDD MP-SPDZ production launch expects exactly three parties")
    retries = min(2, max(0, int(max_transient_retries)))
    port_span = [int(base_port) + offset for offset in range(3)]
    if any(port < 1 or port > 65535 for port in port_span):
        raise ValueError("MP-SPDZ base port must leave room for base_port through base_port + 2")
    attempts: List[Dict[str, Any]] = []
    retry_reasons: List[str] = []
    for attempt_index in range(retries + 1):
        if not wait_for_tcp_ports_release("127.0.0.1", port_span, timeout_seconds=5.0):
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "protocol": "three-party",
                "base_port": int(base_port),
                "checked_ports": port_span,
                "attempt_count": len(attempts),
                "retry_count": max(0, len(attempts) - 1),
                "retry_reasons": retry_reasons,
                "attempts": attempts,
                "status": "failed",
                "failure_class": "port_not_released",
            }
            raise MpSpdzPartyExecutionError(
                f"MP-SPDZ launcher ports {port_span} were not released", receipt
            )
        # Record retries after preflight admits another party launch.
        if attempt_index > 0:
            retry_reasons.append(str(attempts[-1]["failure_class"]))
        output, attempt = _run_party_attempt(
            commands,
            cwd=canonical_mp_spdz_path(mp_spdz_path),
            timeout_seconds=timeout_seconds,
            startup_delay_seconds=startup_delay_seconds,
            failure_grace_seconds=failure_grace_seconds,
            max_stderr_excerpt_chars=max_stderr_excerpt_chars,
        )
        attempt["attempt"] = attempt_index + 1
        attempts.append(attempt)
        if attempt["status"] == "success":
            return output, {
                "schema": RECEIPT_SCHEMA,
                "protocol": "three-party",
                "base_port": int(base_port),
                "attempt_count": len(attempts),
                "retry_count": len(attempts) - 1,
                "retry_reasons": retry_reasons,
                "attempts": attempts,
                "status": "success",
                "failure_class": "none",
            }
        if not attempt["cleanup_complete"]:
            break
        if not attempt["retryable"] or attempt_index >= retries:
            break
        if not wait_for_tcp_ports_release("127.0.0.1", port_span, timeout_seconds=5.0):
            break
        time.sleep(max(0.0, float(retry_backoff_seconds)))

    final = attempts[-1] if attempts else {}
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "protocol": "three-party",
        "base_port": int(base_port),
        "attempt_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "retry_reasons": retry_reasons,
        "attempts": attempts,
        "status": "failed",
        "failure_class": final.get("failure_class", "unknown"),
    }
    diagnostics = "\n".join(
        f"P{row['party']}: {row['excerpt']}"
        for row in final.get("party_stderr", [])
        if row.get("excerpt")
    )
    raise MpSpdzPartyExecutionError(
        "MP-SPDZ party execution failed" + (f"\n{diagnostics}" if diagnostics else ""),
        receipt,
    )

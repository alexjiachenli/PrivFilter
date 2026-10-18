import gc
import time
from typing import Dict, Any, Optional, List


def _is_torch_tensor(obj: Any) -> bool:
    try:
        import torch

        return isinstance(obj, torch.Tensor)
    except Exception:
        return False


def state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new state_dict with all tensors moved to CPU.
    Non-tensor values are kept as-is.
    """
    try:
        import torch

        cpu_sd: Dict[str, Any] = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                cpu_sd[k] = v.detach().to("cpu")
            else:
                cpu_sd[k] = v
        return cpu_sd
    except Exception:
        return state_dict


def clear_cuda_memory(tag: str = None) -> None:
    """Aggressively clear CUDA caches and run Python GC.
    Optionally pass a tag for logging in the caller.
    """
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def free_tensors(*objs: Any) -> None:
    """Delete references to tensors and clear CUDA memory."""
    try:
        for obj in objs:
            try:
                del obj
            except Exception:
                pass
    except Exception:
        pass
    clear_cuda_memory()


def _fmt_bytes(n: float) -> str:
    try:
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        i = 0
        while n >= 1024 and i < len(units) - 1:
            n /= 1024.0
            i += 1
        return f"{n:.1f}{units[i]}"
    except Exception:
        return f"{n}B"


def _system_memory_snapshot() -> Dict[str, Any]:
    info = {
        "total": None,
        "available": None,
        "used": None,
        "used_pct": None,
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["total"] = int(vm.total)
        info["available"] = int(vm.available)
        info["used"] = int(vm.used)
        info["used_pct"] = float(vm.percent)
        return info
    except Exception:
        # Fallback to /proc/meminfo
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) != 2:
                        continue
                    k = parts[0]
                    v = parts[1].strip().split()[0]
                    meminfo[k] = int(v) * 1024  # kB -> bytes
            total = meminfo.get("MemTotal")
            available = meminfo.get("MemAvailable")
            used = total - available if (total is not None and available is not None) else None
            used_pct = (used / total * 100.0) if (used is not None and total) else None
            info.update(
                {"total": total, "available": available, "used": used, "used_pct": used_pct}
            )
        except Exception:
            pass
        return info


def _process_memory_snapshot() -> Dict[str, Any]:
    info = {
        "rss": None,  # Resident Set Size
        "vms": None,  # Virtual Memory Size
    }
    try:
        import psutil  # type: ignore

        p = psutil.Process()
        mi = p.memory_info()
        info["rss"] = int(mi.rss)
        info["vms"] = int(mi.vms)
        return info
    except Exception:
        try:
            import resource

            # ru_maxrss: kilobytes on Linux
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            info["rss"] = int(rss_kb) * 1024
        except Exception:
            pass
        return info


def _gpu_memory_snapshot() -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for idx in range(count):
                try:
                    name = torch.cuda.get_device_name(idx)
                except Exception:
                    name = f"cuda:{idx}"
                try:
                    alloc = torch.cuda.memory_allocated(idx)
                except Exception:
                    alloc = None
                try:
                    reserved = torch.cuda.memory_reserved(idx)
                except Exception:
                    reserved = None
                try:
                    peak = torch.cuda.max_memory_allocated(idx)
                except Exception:
                    peak = None
                gpus.append(
                    {
                        "index": idx,
                        "name": name,
                        "allocated": alloc,
                        "reserved": reserved,
                        "peak_allocated": peak,
                    }
                )
    except Exception:
        pass
    return gpus


def memory_snapshot(tag: Optional[str] = None) -> str:
    """Return a formatted memory snapshot string including system, process and CUDA stats.

    Example:
        logger.info(memory_snapshot("epoch=12/start"))
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    sys_info = _system_memory_snapshot()
    proc_info = _process_memory_snapshot()
    gpu_infos = _gpu_memory_snapshot()

    parts: List[str] = []
    header = f"[mem] ts={ts}"
    if tag:
        header += f" tag={tag}"
    parts.append(header)
    if sys_info["total"] is not None and sys_info["used"] is not None:
        parts.append(f"sys_used={_fmt_bytes(sys_info['used'])}/{_fmt_bytes(sys_info['total'])}")
        if sys_info["used_pct"] is not None:
            parts.append(f"sys_used_pct={sys_info['used_pct']:.1f}%")
    if proc_info["rss"] is not None:
        parts.append(f"proc_rss={_fmt_bytes(proc_info['rss'])}")
    if proc_info["vms"] is not None:
        parts.append(f"proc_vms={_fmt_bytes(proc_info['vms'])}")
    if gpu_infos:
        for gi in gpu_infos:
            idx = gi.get("index")
            name = gi.get("name")
            alloc = gi.get("allocated")
            reserved = gi.get("reserved")
            peak = gi.get("peak_allocated")
            parts.append(
                f"gpu[{idx}:{name}] alloc={_fmt_bytes(alloc) if alloc is not None else 'n/a'} "
                f"reserved={_fmt_bytes(reserved) if reserved is not None else 'n/a'} "
                f"peak_alloc={_fmt_bytes(peak) if peak is not None else 'n/a'}"
            )
    return " | ".join(parts)

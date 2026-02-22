"""Time and memory profiling helpers for scaling benchmarks."""

from __future__ import annotations

from pathlib import Path
import json
import os
import time
from typing import Any, Callable


def time_it(label: str, fn: Callable[[], Any]) -> tuple[Any, float]:
    """Execute ``fn`` and return ``(result, elapsed_seconds)``."""

    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def memory_snapshot() -> dict[str, Any]:
    """Return a best-effort memory snapshot.

    Uses ``psutil`` if available, else ``resource`` on Unix-like systems.
    """

    try:  # pragma: no cover - optional dependency
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        return {
            "backend": "psutil",
            "rss_bytes": int(mem.rss),
            "vms_bytes": int(mem.vms),
        }
    except Exception:
        try:
            import resource  # type: ignore

            usage = resource.getrusage(resource.RUSAGE_SELF)
            # Linux reports KB; macOS reports bytes. Mark unit explicitly.
            return {
                "backend": "resource",
                "ru_maxrss": int(usage.ru_maxrss),
                "ru_maxrss_unit": "KB_on_linux_bytes_on_macos",
            }
        except Exception as exc:  # pragma: no cover
            return {"backend": "unavailable", "error": str(exc)}


def write_profile_report(path: str | Path, payload: dict[str, Any]) -> str:
    """Write a profile report payload as JSON."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(out_path)


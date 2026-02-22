"""
CPU affinity utilities for pinning threads and subprocesses to specific cores.

Linux-only: all functions are safe no-ops on platforms that don't support
os.sched_setaffinity (e.g. macOS, Windows). Designed for edge/big.LITTLE
deployments where separating streaming and inference cores matters.
"""

import os
from typing import List, Optional, Set

from .central_logger import get_logger

_logger = get_logger("cpu_affinity")

_cpu_count: int = os.cpu_count() or 1


def _validate_cores(cores: List[int], label: str) -> bool:
    """Return True if every core index is within [0, cpu_count-1]."""
    invalid = [c for c in cores if not (0 <= c < _cpu_count)]
    if invalid:
        _logger.warning(f"[{label}] Invalid core indices {invalid} (cpu_count={_cpu_count}), skipping affinity")
        return False
    return True


def set_thread_affinity(cores: List[int], label: str = "") -> bool:
    """
    Pin the **calling thread** to the given CPU cores.

    Args:
        cores: List of core indices to allow scheduling on.
        label: Human-readable label used in log messages.

    Returns:
        True on success, False if unsupported or an error occurred.
    """
    if not cores:
        return False
    if not _validate_cores(cores, label):
        return False
    try:
        os.sched_setaffinity(0, cores)  # 0 = current thread
        _logger.info(f"[{label}] Thread affinity set to cores {sorted(cores)}")
        return True
    except AttributeError:
        # os.sched_setaffinity not available on this platform (macOS, Windows)
        _logger.debug(f"[{label}] sched_setaffinity not supported on this platform")
        return False
    except OSError as e:
        _logger.warning(f"[{label}] Failed to set thread affinity: {e}")
        return False


def set_process_affinity(pid: int, cores: List[int], label: str = "") -> bool:
    """
    Pin a **subprocess** to the given CPU cores.

    Calls os.sched_setaffinity from the parent process, so it does not
    conflict with any preexec_fn set on the subprocess.

    Args:
        pid:   PID of the target process.
        cores: List of core indices to allow scheduling on.
        label: Human-readable label used in log messages.

    Returns:
        True on success, False if unsupported or an error occurred.
    """
    if not cores:
        return False
    if not _validate_cores(cores, label):
        return False
    try:
        os.sched_setaffinity(pid, cores)
        _logger.info(f"[{label}] Process {pid} affinity set to cores {sorted(cores)}")
        return True
    except AttributeError:
        _logger.debug(f"[{label}] sched_setaffinity not supported on this platform")
        return False
    except OSError as e:
        _logger.warning(f"[{label}] Failed to set affinity for PID {pid}: {e}")
        return False


def get_current_affinity() -> Optional[Set[int]]:
    """
    Return the CPU affinity mask of the calling thread.

    Returns:
        Set of allowed core indices, or None if unsupported.
    """
    try:
        return os.sched_getaffinity(0)
    except AttributeError:
        return None
    except OSError:
        return None

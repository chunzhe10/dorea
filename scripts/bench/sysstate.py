"""System state capture for bench harness.

Captures hardware/runtime state that affects measurement reproducibility.
Uses pynvml for GPU state (more reliable than parsing nvidia-smi output).
"""

import platform
import sys
import time
from pathlib import Path

import torch

from schema import SystemState


def _read_sysfs(path: str, default: str = "") -> str:
    try:
        return Path(path).read_text().strip()
    except (FileNotFoundError, PermissionError):
        return default


def _cpu_governor() -> str:
    gov = _read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return gov or "unknown"


def _cpu_turbo_enabled() -> bool:
    # Intel pstate: /sys/devices/system/cpu/intel_pstate/no_turbo (0 = turbo on)
    no_turbo = _read_sysfs("/sys/devices/system/cpu/intel_pstate/no_turbo")
    if no_turbo:
        return no_turbo == "0"
    # AMD: /sys/devices/system/cpu/cpufreq/boost (1 = on)
    boost = _read_sysfs("/sys/devices/system/cpu/cpufreq/boost")
    if boost:
        return boost == "1"
    return False


def _thp() -> str:
    return _read_sysfs("/sys/kernel/mm/transparent_hugepage/enabled", "unknown")


def _swappiness() -> int:
    try:
        return int(_read_sysfs("/proc/sys/vm/swappiness", "0"))
    except ValueError:
        return -1


def _kernel_version() -> str:
    return platform.release()


def _in_container() -> bool:
    # Heuristic: /.dockerenv exists, or /proc/1/cgroup mentions docker/containerd
    if Path("/.dockerenv").exists():
        return True
    try:
        cg = Path("/proc/1/cgroup").read_text()
        return "docker" in cg or "containerd" in cg or "kubepods" in cg
    except OSError:
        return False


def _trt_version() -> str | None:
    try:
        import tensorrt
        return tensorrt.__version__
    except ImportError:
        return None


def _pcie_health_gbps() -> float:
    """Measure actual pinned PCIe bandwidth to detect degraded links."""
    size_mb = 100
    n_elem = size_mb * 1024 * 1024 // 4
    src = torch.empty(n_elem, dtype=torch.int32, pin_memory=True)
    src.fill_(1)
    # Warmup
    for _ in range(3):
        _ = src.cuda(non_blocking=False)
    torch.cuda.synchronize()
    n_iters = 10
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = src.cuda(non_blocking=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bytes_transferred = n_iters * size_mb * 1024 * 1024
    return (bytes_transferred / elapsed) / 1e9


def capture_system_state() -> SystemState:
    """Capture full system state. Raises on unrecoverable errors."""
    import pynvml
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name_raw = pynvml.nvmlDeviceGetName(handle)
        gpu_name = name_raw.decode() if isinstance(name_raw, bytes) else name_raw
        # Older pynvml releases lack nvmlDeviceGetCudaComputeCapability; fall back to torch.
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        except AttributeError:
            major, minor = torch.cuda.get_device_capability(0)
        driver_raw = pynvml.nvmlSystemGetDriverVersion()
        driver_version = driver_raw.decode() if isinstance(driver_raw, bytes) else driver_raw
        gfx_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        gfx_clock_max = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        mem_clock_max = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        try:
            power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            power_limit_w = power_limit_mw // 1000
        except pynvml.NVMLError:
            power_limit_w = 0
        try:
            persist_mode = pynvml.nvmlDeviceGetPersistenceMode(handle) == pynvml.NVML_FEATURE_ENABLED
        except pynvml.NVMLError:
            persist_mode = False
        try:
            ecc = pynvml.nvmlDeviceGetEccMode(handle)[0] == pynvml.NVML_FEATURE_ENABLED
        except pynvml.NVMLError:
            ecc = False
        pcie_gen_cur = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
        pcie_width_cur = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
        pcie_gen_max = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
        pcie_width_max = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
    finally:
        pynvml.nvmlShutdown()

    return SystemState(
        kernel_version=_kernel_version(),
        cpu_governor=_cpu_governor(),
        cpu_turbo_enabled=_cpu_turbo_enabled(),
        thp_enabled=_thp(),
        swappiness=_swappiness(),
        gpu_name=gpu_name,
        gpu_compute_cap=f"{major}.{minor}",
        driver_version=driver_version,
        cuda_version=torch.version.cuda or "unknown",
        gpu_clock_graphics_mhz=gfx_clock,
        gpu_clock_memory_mhz=mem_clock,
        gpu_clock_graphics_max_mhz=gfx_clock_max,
        gpu_clock_memory_max_mhz=mem_clock_max,
        gpu_power_limit_w=power_limit_w,
        gpu_persistence_mode=persist_mode,
        gpu_ecc_enabled=ecc,
        pcie_link_gen_current=pcie_gen_cur,
        pcie_link_width_current=pcie_width_cur,
        pcie_link_gen_max=pcie_gen_max,
        pcie_link_width_max=pcie_width_max,
        pcie_health_gbps=_pcie_health_gbps(),
        in_container=_in_container(),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        trt_version=_trt_version(),
    )

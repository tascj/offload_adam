"""Exact-size pinned host allocation with optional NUMA binding.

Each call does mmap + cudaHostRegister, optionally mbind to the NUMA node
of the current CUDA device. This avoids PyTorch's power-of-2 rounding in
its pinned allocator and the cross-socket DMA penalty in multi-GPU runs.

Linux-only. libnuma.so.1 is optional; without it, NUMA binding is skipped.
"""

import ctypes
import os
import weakref
from pathlib import Path

import torch

_PAGE_SIZE = 4096
_MAP_PRIVATE = 0x02
_MAP_ANONYMOUS = 0x20
_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_FAILED = ctypes.c_void_p(-1).value
_MPOL_BIND = 2
_MPOL_MF_STRICT = 1 << 0
_MPOL_MF_MOVE = 1 << 1
# nodemask lives in one c_ulong (64 bits); enough for 64 NUMA nodes.
_MAX_NUMA_NODES = ctypes.sizeof(ctypes.c_ulong) * 8


_cudart_handle = None
_libc_handle = None
_libnuma_sentinel = object()
_libnuma_handle = _libnuma_sentinel


def _cudart():
    global _cudart_handle
    if _cudart_handle is None:
        lib = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
        lib.cudaHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        lib.cudaHostRegister.restype = ctypes.c_int
        lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        lib.cudaHostUnregister.restype = ctypes.c_int
        lib.cudaDeviceGetPCIBusId.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.cudaDeviceGetPCIBusId.restype = ctypes.c_int
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        _cudart_handle = lib
    return _cudart_handle


def _libc():
    global _libc_handle
    if _libc_handle is None:
        lib = ctypes.CDLL("libc.so.6", use_errno=True)
        lib.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        lib.mmap.restype = ctypes.c_void_p
        lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.munmap.restype = ctypes.c_int
        lib.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        lib.memset.restype = ctypes.c_void_p
        _libc_handle = lib
    return _libc_handle


def _libnuma():
    global _libnuma_handle
    if _libnuma_handle is _libnuma_sentinel:
        try:
            lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
        except OSError:
            _libnuma_handle = None
            return None
        lib.mbind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_uint,
        ]
        lib.mbind.restype = ctypes.c_long
        _libnuma_handle = lib
    return _libnuma_handle


def _cuda_check(err):
    if err != 0:
        msg = _cudart().cudaGetErrorString(err).decode()
        raise RuntimeError(f"CUDA runtime error {err}: {msg}")


def gpu_numa_node(device=None) -> int:
    """Return the NUMA node the given CUDA device is attached to, or -1.

    Reads /sys/bus/pci/devices/<pci>/numa_node after resolving the PCI bus id
    via cudaDeviceGetPCIBusId. A value of -1 means the kernel did not
    associate the device with any NUMA node (typical on single-socket hosts).
    """
    if device is None:
        device = torch.cuda.current_device()
    buf = ctypes.create_string_buffer(16)
    _cuda_check(_cudart().cudaDeviceGetPCIBusId(buf, 16, device))
    pci = buf.value.decode().lower()
    try:
        return int(Path(f"/sys/bus/pci/devices/{pci}/numa_node").read_text().strip())
    except (OSError, ValueError):
        return -1


def _alloc(size, numa_target):
    """mmap + optional mbind + first-touch + cudaHostRegister."""
    libc = _libc()
    ptr = libc.mmap(
        None,
        size,
        _PROT_READ | _PROT_WRITE,
        _MAP_PRIVATE | _MAP_ANONYMOUS,
        -1,
        0,
    )
    if ptr is None or ptr == _MAP_FAILED:
        err = ctypes.get_errno()
        raise OSError(err, f"mmap({size}) failed: {os.strerror(err)}")

    if numa_target is not None:
        numa = _libnuma()
        assert numa is not None
        nodemask = (ctypes.c_ulong * 1)(1 << numa_target)
        ret = numa.mbind(
            ptr,
            size,
            _MPOL_BIND,
            nodemask,
            _MAX_NUMA_NODES,
            _MPOL_MF_STRICT | _MPOL_MF_MOVE,
        )
        if ret != 0:
            err = ctypes.get_errno()
            libc.munmap(ptr, size)
            raise OSError(
                err, f"mbind(node={numa_target}) failed: {os.strerror(err)}"
            )

    # First-touch: commit physical pages now so the mbind policy applies,
    # rather than letting them fault in lazily during later access.
    libc.memset(ptr, 0, size)

    try:
        _cuda_check(_cudart().cudaHostRegister(ptr, size, 0))
    except BaseException:
        libc.munmap(ptr, size)
        raise

    cudart_lib = _cudart()
    libc_lib = libc

    def cleanup():
        try:
            cudart_lib.cudaHostUnregister(ptr)
        except Exception:
            pass
        try:
            libc_lib.munmap(ptr, size)
        except Exception:
            pass

    return ptr, cleanup


def zeros_pinned(shape, dtype, numa_node="auto"):
    """Allocate a zero-initialized pinned CPU tensor of exact size.

    Args:
        shape: tuple or torch.Size.
        dtype: torch.dtype.
        numa_node:
            "auto" — bind to the current CUDA device's NUMA node when it
                is known and libnuma is available; otherwise skip binding.
            int    — bind to this specific NUMA node (requires libnuma).
            None   — skip NUMA binding.

    Returns:
        A zero-filled CPU tensor with pinned backing memory. When the last
        reference is dropped, the backing memory is released automatically.
    """
    # Tensor.is_pinned only recognizes cudaHostRegister after the CUDA primary
    # context exists; without this, non_blocking copies silently go sync.
    if torch.cuda.is_available():
        torch.cuda.init()

    shape = torch.Size(shape)
    elem_size = torch.empty(0, dtype=dtype).element_size()
    nbytes = shape.numel() * elem_size
    size = ((nbytes + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE
    if size == 0:
        size = _PAGE_SIZE

    target = None
    if numa_node == "auto":
        if torch.cuda.is_available() and _libnuma() is not None:
            detected = gpu_numa_node()
            if detected >= 0:
                target = detected
    elif isinstance(numa_node, int):
        if _libnuma() is None:
            raise RuntimeError(
                "libnuma.so.1 not available; install libnuma1 or pass "
                "numa_node=None"
            )
        target = numa_node
    elif numa_node is not None:
        raise ValueError(
            f"numa_node must be 'auto', an int, or None (got {numa_node!r})"
        )

    ptr, cleanup = _alloc(size, target)
    try:
        buf = (ctypes.c_uint8 * size).from_address(ptr)
        flat = torch.frombuffer(buf, dtype=torch.uint8)
        weakref.finalize(flat.untyped_storage(), cleanup)
        typed = flat.view(dtype)
        return typed[: shape.numel()].view(shape)
    except BaseException:
        cleanup()
        raise

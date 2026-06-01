"""Preload nvidia/cu13 libraries so PyTorch CUDA JIT can find libnvrtc-builtins.

Must be imported before any torch / transformers CUDA usage.
"""

import ctypes
import os
import sys


def _preload_nvidia_cu13():
    # Find the nvidia/cu13/lib directory inside the active venv
    candidates = []

    # Look inside the current venv
    venv_base = os.environ.get("VIRTUAL_ENV")
    if venv_base:
        candidates.append(
            os.path.join(venv_base, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cu13", "lib")
        )

    # Also check the running interpreter's site-packages
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        candidates.append(os.path.join(sp, "nvidia", "cu13", "lib"))

    for lib_dir in candidates:
        nvrtc = os.path.join(lib_dir, "libnvrtc.so.13")
        builtins = os.path.join(lib_dir, "libnvrtc-builtins.so.13.0")
        if os.path.isfile(nvrtc) and os.path.isfile(builtins):
            try:
                ctypes.CDLL(nvrtc, mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(builtins, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass  # best-effort
            break


_preload_nvidia_cu13()

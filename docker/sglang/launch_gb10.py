#!/usr/bin/env python3
"""
Launch SGLang with GB10/DGX Spark UMA workaround.

DGX Spark uses unified memory (UMA). nvidia-smi reports "Not Supported" for
memory.total, and torch.cuda.mem_get_info() can fail with OOM. This script
patches get_nvgpu_memory_capacity to return a fallback (100GB) when detection fails.

Uses multiprocessing fork (Linux) to avoid spawn/bootstrap issues when run via runpy.
"""
import sys
import os


def _patch_gb10_memory():
    """Patch SGLang memory detection for GB10 UMA before any server code runs."""
    import sglang.srt.utils.common as common

    _orig = common.get_nvgpu_memory_capacity

    def _patched():
        try:
            return _orig()
        except Exception as e:
            # GB10 UMA: mem_get_info/nvidia-smi often fail; use 100GB fallback
            fallback_mb = int(os.environ.get("SGLANG_GB10_MEM_MB", "100000"))
            print(
                f"[sglang-gb10] Using {fallback_mb}MB fallback (GB10 UMA): {e}",
                file=sys.stderr,
            )
            return fallback_mb

    common.get_nvgpu_memory_capacity = _patched


def _main():
    # Use fork on Linux to avoid spawn/bootstrap issues (child re-importing __main__)
    try:
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # Already set
    _patch_gb10_memory()
    import runpy
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    _main()

"""Unit tests for voice STT fatal-CUDA detection + health flag."""
from __future__ import annotations

import importlib


def _fresh_module():
    # Reload so each test starts from a clean (healthy) flag.
    import app.adapters.stt_nemotron as m
    importlib.reload(m)
    return m


def test_healthy_by_default():
    m = _fresh_module()
    assert m.stt_healthy() is True


def test_fatal_cuda_error_flags_unhealthy():
    m = _fresh_module()
    assert m.note_stt_error(RuntimeError("CUDA error: unknown error")) is True
    assert m.stt_healthy() is False
    # idempotent: stays unhealthy
    m.note_stt_error(RuntimeError("CUDA error: again"))
    assert m.stt_healthy() is False


def test_variants_detected():
    for msg in (
        "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED",
        "CUBLAS_STATUS_NOT_INITIALIZED",
        "an illegal memory access was encountered",
        "device-side assert triggered",
    ):
        m = _fresh_module()
        assert m.note_stt_error(RuntimeError(msg)) is True, msg
        assert m.stt_healthy() is False


def test_benign_error_does_not_flag():
    m = _fresh_module()
    assert m.note_stt_error(ValueError("empty audio buffer")) is False
    assert m.note_stt_error(TimeoutError("slow")) is False
    assert m.stt_healthy() is True

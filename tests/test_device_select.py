import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "funasr" / "utils" / "device_select.py"
DOCKER_DIR = ROOT / "runtime" / "dockerfile"


def load_module():
    spec = importlib.util.spec_from_file_location("device_select_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_torch_npu(monkeypatch, available):
    stub = types.ModuleType("torch_npu")
    stub.npu = types.SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, "torch_npu", stub)


def hide_torch_npu(monkeypatch):
    # A None entry makes ``import torch_npu`` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch_npu", None)


def test_default_device_prefers_visible_npu(monkeypatch):
    monkeypatch.delenv("FUNASR_DEVICE", raising=False)
    install_torch_npu(monkeypatch, available=True)

    assert load_module().default_device() == "npu:0"


def test_default_device_keeps_cuda_without_torch_npu(monkeypatch):
    monkeypatch.delenv("FUNASR_DEVICE", raising=False)
    hide_torch_npu(monkeypatch)

    assert load_module().default_device() == "cuda:0"


def test_default_device_keeps_cuda_when_no_npu_visible(monkeypatch):
    monkeypatch.delenv("FUNASR_DEVICE", raising=False)
    install_torch_npu(monkeypatch, available=False)

    assert load_module().default_device() == "cuda:0"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("FUNASR_DEVICE", "npu:3")
    install_torch_npu(monkeypatch, available=True)

    assert load_module().default_device() == "npu:3"


def test_prepare_device_rejects_npu_without_backend(monkeypatch):
    hide_torch_npu(monkeypatch)
    module = load_module()

    with pytest.raises(RuntimeError, match="torch_npu"):
        module.prepare_device("npu:0")
    assert module.prepare_device("cuda:0") == "cuda:0"


def test_npu_dockerfile_uses_switchable_vllm_ascend_base():
    dockerfile = (DOCKER_DIR / "Dockerfile.npu").read_text()
    compose = (DOCKER_DIR / "docker-compose.npu.yml").read_text()

    assert "ARG VLLM_ASCEND_IMAGE=quay.io/ascend/vllm-ascend:" in dockerfile
    assert "FROM ${VLLM_ASCEND_IMAGE}" in dockerfile
    assert "-c /tmp/base-constraints.txt" in dockerfile
    assert "EXPOSE 8899 10095" in dockerfile
    assert "VLLM_ASCEND_TAG" in compose
    assert "/dev/davinci_manager" in compose


def test_npu_entrypoint_starts_openai_and_realtime_services():
    entrypoint = (DOCKER_DIR / "npu-entrypoint.sh").read_text()

    for marker in (
        "set -euo pipefail",
        "serve_vllm.py",
        "funasr.bin.realtime_ws",
        "FUNASR_SERVICE",
        "wait -n",
    ):
        assert marker in entrypoint

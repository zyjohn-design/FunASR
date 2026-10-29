# Copyright FunASR (https://github.com/modelscope/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""Default device selection for service entry points (CUDA and Ascend NPU).

Kept free of top-level torch imports so CLI parsers and tests stay lightweight.
"""

import importlib
import logging
import os

logger = logging.getLogger(__name__)

DEVICE_ENV = "FUNASR_DEVICE"
_FALLBACK_DEVICE = "cuda:0"


def ensure_npu_backend():
    """Import torch_npu so the ``npu`` device is registered with torch.

    Returns True when torch_npu imports and reports at least one usable NPU.
    """
    try:
        torch_npu = importlib.import_module("torch_npu")
    except ImportError:
        return False
    except Exception as error:  # broken CANN/driver installs raise non-ImportError
        logger.warning("torch_npu import failed: %s", error)
        return False
    try:
        return bool(torch_npu.npu.is_available())
    except Exception as error:
        logger.warning("torch_npu.npu.is_available() failed: %s", error)
        return False


def default_device():
    """Return the default inference device for service CLIs.

    Order: ``$FUNASR_DEVICE`` override, then ``npu:0`` when torch_npu sees an NPU,
    otherwise ``cuda:0`` (the historical default; AutoModel falls back to CPU).
    """
    override = os.environ.get(DEVICE_ENV, "").strip()
    if override:
        if override.startswith("npu"):
            ensure_npu_backend()
        return override
    if ensure_npu_backend():
        return "npu:0"
    return _FALLBACK_DEVICE


def prepare_device(device):
    """Validate the chosen device before models load; register NPU if needed."""
    if str(device).startswith("npu") and not ensure_npu_backend():
        raise RuntimeError(
            f"device={device!r} requested, but torch_npu is not installed or no NPU "
            "is visible. Check CANN/torch_npu and the /dev/davinci* device mapping."
        )
    return device

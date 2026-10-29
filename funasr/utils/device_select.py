# Copyright FunASR (https://github.com/modelscope/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""Default device selection for service entry points (CUDA and Ascend NPU).

Backend probing is shared with training via ``funasr.train_utils.device_funcs``.
"""

import os

from funasr.train_utils.device_funcs import is_accelerator_available

DEVICE_ENV = "FUNASR_DEVICE"
_FALLBACK_DEVICE = "cuda:0"


def default_device():
    """Return the default inference device for service CLIs.

    Order: ``$FUNASR_DEVICE`` override, then ``npu:0`` when torch_npu sees an NPU,
    otherwise ``cuda:0`` (the historical default; AutoModel falls back to CPU).
    Probing NPU imports torch_npu, which registers the ``npu`` device with torch.
    """
    override = os.environ.get(DEVICE_ENV, "").strip()
    if override:
        if override.startswith("npu"):
            is_accelerator_available("npu")
        return override
    if is_accelerator_available("npu"):
        return "npu:0"
    return _FALLBACK_DEVICE


def prepare_device(device):
    """Validate the chosen device before models load; register NPU if needed."""
    if str(device).startswith("npu") and not is_accelerator_available("npu"):
        raise RuntimeError(
            f"device={device!r} requested, but torch_npu is not installed or no NPU "
            "is visible. Check CANN/torch_npu and the /dev/davinci* device mapping."
        )
    return device

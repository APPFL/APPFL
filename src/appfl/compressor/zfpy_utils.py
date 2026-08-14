"""
Tolerant loading of the optional ``zfpy`` dependency.

``zfpy`` is not installed as part of the base APPFL requirements: its wheels are
built against the numpy 1.x ABI and fail to import under ``numpy>=2.0.0``.
Pinning numpy for every APPFL user just to keep one optional lossy compressor
working is too restrictive, so ZFP support is opt-in::

    pip install "appfl[zfp]"

which installs ``zfpy`` together with a compatible ``numpy<2.0.0``.
"""

import numpy as np

try:
    import zfpy

    _ZFP_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - a numpy ABI mismatch may raise anything
    zfpy = None
    _ZFP_IMPORT_ERROR = e

ZFP_COMPATIBLE = zfpy is not None


def _numpy_major_version() -> int:
    try:
        return int(np.__version__.split(".")[0])
    except (ValueError, IndexError):
        return 0


def require_zfpy():
    """
    Return the imported ``zfpy`` module, or raise an actionable ``ImportError``
    explaining how to make ZFP available.
    """
    if zfpy is not None:
        return zfpy
    if _numpy_major_version() >= 2:
        reason = (
            f"zfpy is not usable with your numpy version ({np.__version__}) - "
            "it requires numpy<2.0.0."
        )
    else:
        reason = "zfpy is not installed."
    raise ImportError(
        f"The ZFP compressor is unavailable: {reason} "
        'Install ZFP support with `pip install "appfl[zfp]"` (this pins '
        "numpy<2.0.0), or use another compressor such as SZ2Compressor, "
        "SZ3Compressor, or SZxCompressor, which work with numpy 2.x."
    ) from _ZFP_IMPORT_ERROR

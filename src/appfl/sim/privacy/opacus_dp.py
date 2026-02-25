# This module has moved to appfl.privacy.
# Re-exported here for backward compatibility.
try:
    from appfl.privacy.opacus_dp import make_private_with_opacus  # noqa: F401
except Exception:  # pragma: no cover
    make_private_with_opacus = None

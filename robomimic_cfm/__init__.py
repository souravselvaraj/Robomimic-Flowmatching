"""
robomimic-cfm — Conditional Flow Matching policy for robomimic.

Importing this package registers the ``"flow_matching"`` algorithm and its config
with robomimic's factories (import-time registration), so downstream code only
needs::

    import robomimic_cfm  # noqa: F401  (registers "flow_matching")

after which ``algo_name="flow_matching"`` works with robomimic's
``config_factory`` / ``algo_factory`` exactly like a built-in algorithm.
"""

# Importing these modules triggers their registration decorators / config
# subclass registration against robomimic. Keep the imports even though the
# names are not used directly here.
from robomimic_cfm import config as _config  # noqa: F401
from robomimic_cfm import flow_matching as _flow_matching  # noqa: F401

from robomimic_cfm.config import FlowMatchingConfig
from robomimic_cfm.flow_matching import FlowMatchingUNet, algo_config_to_class

__version__ = "0.1.0"

__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingUNet",
    "algo_config_to_class",
]

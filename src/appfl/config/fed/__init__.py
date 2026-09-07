"""
Configuration for each algorithm
"""

from .fedasync import FedAsync
from .federated import Federated
from .iceadmm import ICEADMM
from .iiadmm import IIADMM

__all__ = ["ICEADMM", "IIADMM", "FedAsync", "Federated"]

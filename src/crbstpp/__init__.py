"""CRBS-TPP public API.

The package intentionally has no dependency on the archived ``certscr``
implementation.  Only the small, stable types below are public.
"""

from .config import RunConfig
from .data import Dataset
from .report import Certificate, RunReport
from .rules import RuleIdentity, Support
from .solver import FitResult

__all__ = [
    "Certificate",
    "Dataset",
    "FitResult",
    "RuleIdentity",
    "RunConfig",
    "RunReport",
    "Support",
]

__version__ = "0.1.0"


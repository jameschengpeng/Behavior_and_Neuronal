"""Evaluation and testing utilities."""

from .test import test_model
from .metrics import tune_per_class_thresholds
from .postprocessing import hysteresis_decode
from .visualization import plot_ethogram_pair

__all__ = [
    'test_model',
    'tune_per_class_thresholds',
    'hysteresis_decode',
    'plot_ethogram_pair',
]

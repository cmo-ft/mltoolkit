import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections import namedtuple
import logging
from typing import Tuple

log = logging.getLogger(__name__)


class BaseMetric(ABC):
    def __init__(self, output: torch.Tensor, truth_label: torch.Tensor, weight: torch.Tensor) -> None:
        self.metrics: dict[str, float] = self.compute_metrics(output, truth_label, weight)
    
    @abstractmethod
    def compute_metrics(self, prediction, truth_label, weight)->dict[str, float]:
        ...

    @abstractmethod
    def __le__(self, other):
        ...

    @abstractmethod
    def __lt__(self, other):
        ...

    def get_metric_keys(self)->list[str]:
        return self.metrics.keys()
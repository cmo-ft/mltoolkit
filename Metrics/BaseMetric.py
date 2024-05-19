import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections import namedtuple
import logging
from typing import Tuple

log = logging.getLogger(__name__)


class BaseMetric(ABC):
    """
    Base class for defining metrics in a machine learning toolkit.

    Args:
        output (torch.Tensor): The predicted output tensor.
        truth_label (torch.Tensor): The ground truth label tensor.
        weight (torch.Tensor): The weight tensor.

    Attributes:
        metrics (dict[str, float]): A dictionary containing the computed metrics.

    Methods:
        compute_metrics(prediction, truth_label, weight): Abstract method to compute the metrics.
        __le__(other): Abstract method to compare the metric with another metric.
        __lt__(other): Abstract method to compare the metric with another metric.
        get_metric_keys(): Returns a list of metric keys.

    """
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
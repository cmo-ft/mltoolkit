from abc import ABC
from typing import Tuple
import torch
import torch.nn as nn
from collections import namedtuple
import logging
from importlib import import_module
from Metrics.BaseMetric import BaseMetric

log = logging.getLogger(__name__)

class MetricInterface(ABC):
    def __init__(self, metric_config) -> None:
        self.metric_config = metric_config

    def setup(self, metric_config):
        loss_function_name = metric_config['loss_function']
        self.loss_func = getattr(nn, loss_function_name)(reduction='none')

        metric_class_name = self.metric_config.get('metric').split('.') # e.g. 'ClassificationMetric.ClassificationMetric'
        self._metric = getattr(import_module(".".join(['Metrics'] + metric_class_name[:-1])), metric_class_name[-1])
    
    def get_metric_keys(self)->list[str]:
        return self._metric.get_metric_keys()

    def __call__(self, output: torch.Tensor, truth_label: torch.Tensor, weight=None) -> Tuple[torch.float, BaseMetric]:
        """
        Evaluates the model's output and computes the loss and metrics.

        Args:
            output (torch.Tensor): The model's output.
            truth_label (torch.Tensor): The ground truth labels.
            weight (torch.Tensor, optional): The weight for each sample. Defaults to None.

        Returns:
            Tuple[torch.float, torch.float]: A tuple containing the loss and metrics.

        Raises:
            ValueError: If NaN values are present in the output.

        """
        if torch.isnan(output).sum():
            log.error(f"Error: NaN occurred in output: {output}")

        if weight is None:
            weight = torch.ones(len(output))

        loss = (self.loss_func(output, truth_label) * weight).sum() / weight.sum()
        metrics = self._metric(output, truth_label, weight)

        return loss, metrics

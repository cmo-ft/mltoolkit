import torch
from Metrics.BaseMetric import BaseMetric


class RegressionMetric(BaseMetric):
    def __init__(self, output, truth_label, weight) -> None:
        super().__init__(output, truth_label, weight)

    @classmethod
    def get_metric_keys(cls)->list[str]:
        return ['rel_error']

    def compute_metrics(self, output: torch.Tensor, truth_label: torch.Tensor, weight: torch.Tensor)->dict[str, float]:
        rel_error = -torch.abs(output - truth_label) / truth_label
        return {'rel_error': rel_error.mean().item()}

    def __le__(self, other):
        return self.metrics['rel_error'] <= other.metrics['rel_error']
    
    def __lt__(self, other):
        return self.metrics['rel_error'] < other.metrics['rel_error']
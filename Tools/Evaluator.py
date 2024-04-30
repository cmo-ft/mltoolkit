import torch
import torch.nn as nn
from collections import namedtuple


class Evaluator():
    def __init__(self, evaluator_config) -> None:
        self.evaluator_config = evaluator_config
        # criteria for evaluation
        self.Critria = namedtuple('Criteria', ['loss', 'accuracy'])
        self.load()

    def load(self):
        self.loss_func = getattr(nn, self.evaluator_config['loss_function'])(reduction=None)

    """
    Evaluate output of neural network
    """
    def __call__(self, output, truth_label, weight=None):
        loss = (self.loss_func(output, truth_label) * (1 if weight is None else weight)).mean()
        prediction = self.manage_output(output=output)
        accuracy = ((prediction==truth_label) * (1 if weight is None else weight)).mean()
        return self.Critria(loss, accuracy)
    
    """
    Turn output into prediction
    """
    @staticmethod
    def manage_output(output):
        return torch.argmax(output, dim=1)
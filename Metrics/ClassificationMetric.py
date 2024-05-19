from Metrics.BaseMetric import BaseMetric


class ClassificationMetric(BaseMetric):
    def __init__(self, output, truth_label, weight) -> None:
        super().__init__(output, truth_label, weight)
        self.metrics = self.compute_metrics(output, truth_label, weight)

    def compute_metrics(self, prediction, truth_label, weight)->dict[str, float]:
        tp = (prediction * truth_label).sum()
        fp = (prediction * (1 - truth_label)).sum()
        fn = ((1 - prediction) * truth_label).sum()
        tn = ((1 - prediction) * (1 - truth_label)).sum()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        return {'accuracy': accuracy.item(), 'precision': precision.item(), 'recall': recall.item(), 'f1': f1.item()}

    def __le__(self, other):
        return self.metrics['accuracy'] <= other.metrics['accuracy']
    
    def __lt__(self, other):
        return self.metrics['accuracy'] < other.metrics['accuracy']

from typing import List, Dict
from emetrics.metrics import Metric


class MetricsLogger:
    def __init__(self, loss: Metric, metrics: List[Metric]):
        self.metrics = {'train':{metric.m_name: [] for metric in metrics},
                        'eval': {metric.m_name: [] for metric in metrics}}
        self.metrics['train'][loss.m_name] = []
        self.metrics['eval'][loss.m_name] = []

    def __call__(self, metrics_vals: Dict[str, float], phase: str = 'train'):
        for metric_name, metric_val in metrics_vals.items():
            self.metrics[phase][metric_name].append(metric_val)

    def get_logged(self, reformat: bool = True) -> Dict:
        if reformat:
            metrics = {}
            for phase in self.metrics.keys():
                for metric_name, vals in self.metrics[phase].items():
                    col = phase + '_' + metric_name
                    metrics[col] = vals
            return metrics

        return self.metrics
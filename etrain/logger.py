import numpy as np

from emetrics.metrics import Metric, MSE


def psnr_compute(mse: float, max_val: float = 1.0):
        return 10 * np.log10(max_val**2 / mse)


class MetricsLogger:
    def __init__(self, loss: Metric, metrics: list[Metric]):
        self._init_avg_stats(loss, metrics)
        self._init_step_stas()

    def __call__(self, metrics_vals: dict[str, float], phase: str = 'train'):
        for metric_name, metric_val in metrics_vals.items():
            self.metrics[phase][metric_name].append(metric_val)

    def _init_avg_stats(self, loss: Metric, metrics: list[Metric]):
        all_metrics = [loss] + metrics
        self.metrics = {'train': {metric.m_name: [] for metric in all_metrics},
                        'eval': {metric.m_name: [] for metric in all_metrics}}

    def _init_step_stas(self):
        self._step_metrics = {}
        for metric_k in self.metrics['train']:
            self._step_metrics[metric_k] = []

        if 'psnr' in self.metrics['train'].keys():
            self._init_for_psnr()

    def _init_for_psnr(self):
        self._step_metrics[MSE.m_name] = []

    def reinit_step_stats(self):
        self._init_step_stas()

    def update_step_metric_val(self, metric_name: str, metric_val: float):
        self._step_metrics[metric_name].append(metric_val)

    def get_curr_step_metric_val(self, metric_name: str):
        return self._step_metrics[metric_name][-1]

    def get_curr_step_metrics(self):
        return {mkey: f'{self._step_metrics[mkey][-1]:.4f}' for mkey in self._step_metrics}

    def get_avg_metric_val(self, metric_name: str):
        if metric_name == 'psnr':
            return psnr_compute(np.mean(self._step_metrics['mse']))
        return np.mean(self._step_metrics[metric_name])

    def get_avg_metrics(self):
        avg_metrics = {}
        for metric_name in self._step_metrics:
            avg_metrics[metric_name] = self.get_avg_metric_val(metric_name)

    def get_logged(self, reformat: bool = True) -> dict:
        if reformat:
            metrics = {}
            for phase in self.metrics.keys():
                for metric_name, vals in self.metrics[phase].items():
                    col = phase + '_' + metric_name
                    metrics[col] = vals
            return metrics

        return self.metrics
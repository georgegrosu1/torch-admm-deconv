import torch
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader

from emetrics.metrics import Metric


class NNTrainer:
    def __init__(self,
                 loss: Metric,
                 metrics: List[Metric],
                 # saver,
                 # logger
                 ):

        self.loss = loss
        self.metrics = metrics
        # self.saver = saver
        # self.logger = logger
        self._init_stats()


    def _init_stats(self):
        self._metrics_stats = {self.loss.m_name: {'curr': 0.0, 'cumsum': 0.0}}
        if self.metrics:
            for metric_k in self.metrics:
                self._metrics_stats[metric_k.m_name] = {'curr': 0.0, 'cumsum': 0.0}


    def run(self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_dataloader: DataLoader,
            eval_dataloader: DataLoader = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None):

        self.train(model, train_dataloader, optimizer)
        if eval_dataloader:
            self.eval(model, eval_dataloader)


    def train(self, model: torch.nn.Module, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        self._init_stats()
        model.train(mode=True)
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        print('\n\n\n////////////////////////////////////// TRAINING //////////////////////////////////////')
        for batch_idx, (inputs, labels) in pbar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            optimizer.step()

            self._update_performance_stats(loss, outputs, labels)
            self._print_current_metrics(pbar)

        self._print_epoch_metrics('train', len(train_dataloader))

    def _update_performance_stats(self, loss_res, outputs, labels):
        self._metrics_stats[self.loss.m_name]['curr'] = loss_res.item()
        self._metrics_stats[self.loss.m_name]['cumsum'] += loss_res.item()
        for metric_func in self.metrics:
            metric_res = metric_func(outputs, labels).item()
            self._metrics_stats[metric_func.m_name]['curr'] = metric_res
            self._metrics_stats[metric_func.m_name]['cumsum'] += metric_res


    def _print_current_metrics(self, pbar: tqdm):
        metrics_msg = {mkey: f'{self._metrics_stats[mkey]['curr']:.4f}' for mkey in self._metrics_stats}
        pbar.set_postfix(metrics_msg)


    def _print_epoch_metrics(self, phase: str, epoch_steps: int):
        metrics_msg = ''
        for metric_stat in self._metrics_stats.keys():
            avg_v = self._metrics_stats[metric_stat]['cumsum'] / epoch_steps
            metrics_msg += f'{phase}_{metric_stat}: {avg_v:.4f}; '
        print(metrics_msg)


    def eval(self, model: torch.nn.Module, eval_dataloader: DataLoader):
        self._init_stats()
        model.eval()

        print('\n////////////////////////////////////// EVALUATING //////////////////////////////////////')
        with torch.no_grad():
            for batch_idx, (inputs, labels) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                outputs = model(inputs)
                loss = self.loss(outputs, labels)
                self._update_performance_stats(loss, outputs, labels)

        self._print_epoch_metrics('eval', len(eval_dataloader))


    def on_epoch_end(self):
        pass
import torch
from tqdm import tqdm
from typing import List
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from emetrics.metrics import Metric
from etrain.saver import NNSaver
from etrain.logger import MetricsLogger


class NNTrainer:
    def __init__(self,
                 loss: Metric,
                 metrics: List[Metric],
                 saver: NNSaver,
                 logger: MetricsLogger = None):

        self.loss = loss
        self.metrics = metrics
        self.saver = saver
        self.logger = logger
        self._init_stats()


    def _init_stats(self):
        self._metrics_stats = {self.loss.m_name: {'curr': 0.0, 'cumsum': 0.0}}
        if self.metrics:
            for metric_k in self.metrics:
                self._metrics_stats[metric_k.m_name] = {'curr': 0.0, 'cumsum': 0.0}


    def run(self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            train_dataloader: DataLoader,
            eval_dataloader: DataLoader = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None):

        self.get_model_params(model)
        for epoch in range(epochs):
            print(f'\n\n\n/////////////////////////////////// [ EPOCH: {epoch} ] ///////////////////////////////////')
            self.train(model, train_dataloader, optimizer)
            loss_val_steps = len(train_dataloader)
            if eval_dataloader:
                self.eval(model, eval_dataloader, lr_scheduler)
                loss_val_steps = len(eval_dataloader)
            self.on_epoch_end(epoch, model, optimizer, self.get_epoch_metrics(loss_val_steps)[self.loss.m_name])


    def train(self, model: torch.nn.Module, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        self._init_stats()
        model.train(mode=True)
        print('\n [ TRAINING ]')
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, (inputs, labels) in pbar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            optimizer.step()

            self._update_performance_stats(loss, outputs, labels)
            self._print_current_metrics(pbar)

        self._print_epoch_metrics('train', len(train_dataloader))
        self.logger(self.get_epoch_metrics(len(train_dataloader)), 'train')


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
        metrics_msg = '\n'
        for metric_stat, val in self.get_epoch_metrics(epoch_steps).items():
            metrics_msg += f'{phase}_{metric_stat}: {val:.4f}; '
        print(metrics_msg)


    def get_epoch_metrics(self, epoch_steps: int):
        ep_metrics = {}
        for metric_name in self._metrics_stats.keys():
            avg_v = self._metrics_stats[metric_name]['cumsum'] / epoch_steps
            ep_metrics[metric_name] = avg_v
        return ep_metrics


    def eval(self,
             model: torch.nn.Module,
             eval_dataloader: DataLoader,
             lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        self._init_stats()
        model.eval()

        print('\n [ EVALUATING ]')
        with torch.no_grad():
            for batch_idx, (inputs, labels) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                outputs = model(inputs)
                vloss = self.loss(outputs, labels)
                if lr_scheduler is not None:
                    lr_scheduler.step(vloss)
                self._update_performance_stats(vloss, outputs, labels)

        self._print_epoch_metrics('eval', len(eval_dataloader))
        self.logger(self.get_epoch_metrics(len(eval_dataloader)), 'eval')


    @staticmethod
    def get_model_params(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params


    def on_epoch_end(self, epoch, model, optimizer, loss_val):
        logs_dict = self.logger.get_logged(reformat=True) if self.logger is not None else None
        self.saver.save_on_epoch_end(epoch, model, optimizer, loss_val, logs_dict)
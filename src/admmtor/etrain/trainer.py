import torch
from tqdm import tqdm
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from admmtor.emetrics.metrics import Metric, MSE
from admmtor.etrain.saver import NNSaver
from admmtor.etrain.logger import MetricsLogger


class NNTrainer:
    def __init__(self,
                 loss: Metric,
                 metrics: list[Metric],
                 saver: NNSaver,
                 logger: MetricsLogger = None):

        self.loss = loss
        self.saver = saver
        self.logger = logger
        self._init_metrics(metrics)


    def _init_metrics(self, metrics: list[Metric]):
        metrics_names = [metric.m_name for metric in metrics]
        if 'psnr' in metrics_names and 'mse' not in metrics_names:
            self.metrics = metrics + [MSE(metrics[0].device)]
        else:
            self.metrics = metrics


    def run(self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            train_dataloader: DataLoader,
            eval_dataloader: DataLoader = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        
        # Run dummy forward to initialize lazy modules
        dummy_input = torch.randn(1, 3, 64, 64)
        model(dummy_input)

        self.get_model_params(model)
        for epoch in range(epochs):
            print(f'\n\n\n/////////////////////////////////// [ EPOCH: {epoch} ] ///////////////////////////////////')
            self.train(model, train_dataloader, optimizer)
            if eval_dataloader:
                self.eval(model, eval_dataloader, lr_scheduler)
            self.on_epoch_end(epoch, model, optimizer, self.get_epoch_metrics('eval')[self.loss.m_name])


    def train(self, model: torch.nn.Module, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        self.logger.reinit_step_stats()
        model.train(mode=True)
        print('\n [ TRAINING ]')
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, (inputs, labels) in pbar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
            optimizer.step()

            self._update_performance_stats(loss, outputs, labels)
            self._print_current_metrics(pbar)

        self.logger('train')
        self._print_epoch_metrics('train')


    def _update_performance_stats(self, loss_res, outputs, labels):
        self.logger.update_step_metric_val(self.loss.m_name, loss_res.item())
        for metric_func in self.metrics:
            metric_res = metric_func(outputs, labels).item()
            self.logger.update_step_metric_val(metric_func.m_name, metric_res)


    def _print_current_metrics(self, pbar: tqdm):
        metrics_msg = self.logger.get_curr_step_metrics()
        pbar.set_postfix(metrics_msg)


    def _print_epoch_metrics(self, phase: str):
        metrics_msg = '\n'
        for metric_stat, val in self.get_epoch_metrics(phase).items():
            metrics_msg += f'{phase}_{metric_stat}: {val:.4f}; '
        print(metrics_msg)


    def get_epoch_metrics(self, phase: str):
        return self.logger.get_avg_metrics(phase)


    def eval(self,
             model: torch.nn.Module,
             eval_dataloader: DataLoader,
             lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        self.logger.reinit_step_stats()
        model.eval()

        print('\n [ EVALUATING ]')
        with torch.no_grad():
            for batch_idx, (inputs, labels) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                outputs = model(inputs)
                vloss = self.loss(outputs, labels)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                self._update_performance_stats(vloss, outputs, labels)

        self.logger('eval')
        self._print_epoch_metrics('eval')


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
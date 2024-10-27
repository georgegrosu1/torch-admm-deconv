import torch
import numpy as np
import pandas as pd
from utils.train_utils import get_saving_model_path, get_time_formated
from typing import Dict
from enum import Enum


class SaveMode(Enum):
    Each = 0
    Best = 1


class NNSaver:
    def __init__(self, save_dir: str, model_name: str, save_mode: SaveMode = SaveMode.Best, use_time_date: bool = True):
        self.save_dir = save_dir
        self.model_name = model_name
        self.save_mode = save_mode
        save_time = None if not use_time_date else get_time_formated()
        self.model_saving_path = get_saving_model_path(save_dir, model_name, save_time)
        self._losses = np.array([])


    def save_on_epoch_end(self, epoch: int, model: torch.nn.Module , optimizer, vloss: float, log_metrics: Dict = None):
        if self.save_mode == SaveMode.Each:
            self.save_model(epoch, model, optimizer, vloss)
        elif self.save_mode == SaveMode.Best:
            self.save_if_best(epoch, model, optimizer, vloss)
        else:
            raise NotImplementedError

        if log_metrics:
            csv_path = self.model_saving_path.parent / 'logged_metrics.csv'
            pd.DataFrame(log_metrics).to_csv(csv_path)


    def save_if_best(self, epoch: int, model: torch.nn.Module , optimizer, vloss: float):
        if self._losses.size == 0:
            self.save_model(epoch, model, optimizer, vloss)
        else:
            greater_losses = self._losses > vloss
            if greater_losses.sum() == self._losses.shape[0]:
                self.save_model(epoch, model, optimizer, vloss)
        self._losses = np.append(self._losses, vloss)


    def save_model(self, epoch: int, model: torch.nn.Module , optimizer, vloss: float):
        model_path = str(self.model_saving_path).format(epoch=epoch, val_loss=vloss) + '.tar'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': vloss
        }, model_path)

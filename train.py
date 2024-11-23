import os
import cv2
import json
import random
import argparse
from pathlib import Path

import torch
import numpy as np

from eprocessing.dataload import ImageDataset
from modelbuild.restorer import Restorer
from eprocessing.etransforms import Scale, RandCrop, AddAWGN
from etrain.trainer import NNTrainer
from etrain.logger import MetricsLogger
from etrain.saver import NNSaver
from emetrics.metrics import *


def seed_everything(seed=42):
    random.seed(seed)
    cv2.setRNGSeed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.RandomState(seed=seed)
    torch.manual_seed(seed)


def init_training(config_file: str, min_std: int, max_std: int, save_dir: str, model_name: str, device: str):
    config_file_path = os.getcwd() + f'/{config_file}'
    with open(config_file_path, 'r') as f:
        train_cfg = json.load(f)

    # Prepare train & eval data loaders
    im_shape = tuple(train_cfg['im_shape'])
    transforms = [RandCrop(im_shape), Scale()]
    if max_std > 0: transforms += [AddAWGN(std_range=(min_std, max_std), both=False)]
    train_dset = ImageDataset(Path(train_cfg['train']['x_path']), Path(train_cfg['train']['y_path']), device=device,
                              transforms=transforms)
    eval_dset = ImageDataset(Path(train_cfg['eval']['x_path']), Path(train_cfg['eval']['y_path']), device=device,
                             transforms=transforms)
    train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=train_cfg['batch_size'])
    eval_loader = torch.utils.data.DataLoader(eval_dset, shuffle=True, batch_size=train_cfg['batch_size'])

    save_dir_path = os.getcwd() + f'/{save_dir}'
    net_saver = NNSaver(save_dir_path, model_name)

    autoenc_args = {'in_channels': 9,
                    'enc_out_channels': [16, 32, 32],
                    'dec_out_channels': [32, 32, 64],
                    'kernel_sizes': [5, 11, 11],
                    'activation': torch.nn.ReLU6(),
                    'pool_size': 5}
    updowns_args = {'in_channels': 9,
                    'out_channels': [16, 32, 32, 64],
                    'kernel_sizes': [5, 7, 7, 11],
                    'activation': torch.nn.ReLU6()}
    deconv_args_list = [
        {'kern_size': (),
        'max_iters': 80,
         'rho': 0.2,
        'iso': True},
        {'kern_size': (),
         'max_iters': 80,
         'rho': 0.02,
         'iso': False},
        {'kern_size': (),
         'max_iters': 80,
         'rho': 0.004,
         'iso': True},
    ]
    model = Restorer(3, autoenc_args, updowns_args, deconv_args_list)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), train_cfg['lr'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.93)

    eval_metrics = [MSSSIMMetric(device), PSNRMetric(device), SCCMetric(device)]
    loss_func = SSIMLoss(device)

    metrics_logger = MetricsLogger(loss_func, eval_metrics)
    net_trainer = NNTrainer(loss_func, eval_metrics, net_saver, metrics_logger)

    net_trainer.run(model, opt, train_cfg['epochs'], train_loader, eval_loader, lr_scheduler=lr_scheduler)


def main():
    seed_everything()

    args_parser = argparse.ArgumentParser(description='Training script for image restoration')
    args_parser.add_argument('--config_file', '-c', type=str, help='Path to train config file',
                             default=r'configs/train_cfg.json')
    args_parser.add_argument('--min_awgn', '-m', type=int, help='Min std for AWGN',
                             default=0)
    args_parser.add_argument('--max_awgn', '-M', type=int, help='Max std for AWGN',
                             default=0)
    args_parser.add_argument('--save_dir', '-s', type=str, help='Dir (relative to cwd) to save models',
                             default=r'trained_models')
    args_parser.add_argument('--model_name', '-n', type=str, help='Name of the training model',
                             default=r'image_restorer')
    args_parser.add_argument('--device', '-d', type=str, help='Training device (cuda | cpu)',
                            default='cuda')
    args = args_parser.parse_args()

    init_training(args.config_file, args.min_awgn, args.max_awgn, args.save_dir, args.model_name, args.device)


if __name__ == "__main__":
    main()

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
from eprocessing.etransforms import Scale, RandCrop
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


def init_training(config_file, save_dir, model_name, device):
    config_file_path = os.getcwd() + f'/{config_file}'
    with open(config_file_path, 'r') as f:
        train_cfg = json.load(f)

    # Prepare train & eval data loaders
    im_shape = tuple(train_cfg['im_shape'])
    transforms = [RandCrop(im_shape), Scale()]
    train_dset = ImageDataset(Path(train_cfg['train']['x_path']), Path(train_cfg['train']['y_path']), device=device,
                              transforms=transforms)
    eval_dset = ImageDataset(Path(train_cfg['eval']['x_path']), Path(train_cfg['eval']['y_path']), device=device,
                             transforms=transforms)
    train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=train_cfg['batch_size'])
    eval_loader = torch.utils.data.DataLoader(eval_dset, shuffle=True, batch_size=train_cfg['batch_size'])

    save_dir_path = os.getcwd() + f'/{save_dir}'
    net_saver = NNSaver(save_dir_path, model_name)

    autoenc_args = {'in_channels': 3,
                    'enc_out_channels': [16, 32, 32, 64],
                    'dec_out_channels': [32, 32, 64, 64],
                    'kernel_sizes': [5, 11, 11, 15],
                    'activation': torch.nn.ReLU(),
                    'pool_size': 3}
    updowns_args = {'in_channels': 3,
                    'out_channels': [16, 32, 32, 64],
                    'kernel_sizes': [3, 5, 5, 7],
                    'activation': torch.nn.ReLU()}
    deconv_args_list = [
        {'kern_size': (5,5),
        'max_iters': 100,
        'iso': True}
    ]
    model = Restorer(autoenc_args, updowns_args, deconv_args_list)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), train_cfg['lr'])

    eval_metrics = [MSSSIMMetric(device), PSNRMetric(device), SCCMetric(device)]
    loss_func = SSIMLoss(device)

    metrics_logger = MetricsLogger(loss_func, eval_metrics)
    net_trainer = NNTrainer(loss_func, eval_metrics, net_saver, metrics_logger)

    net_trainer.run(model, opt, train_cfg['epochs'], train_loader, eval_loader)


def main():
    seed_everything()

    args_parser = argparse.ArgumentParser(description='Training script for image restoration')
    args_parser.add_argument('--config_file', '-c', type=str, help='Path to train config file',
                             default=r'configs/train_cfg.json')
    args_parser.add_argument('--save_dir', '-s', type=str, help='Dir (relative to cwd) to save models',
                             default=r'trained_models')
    args_parser.add_argument('--model_name', '-n', type=str, help='Name of the training model',
                             default=r'image_restorer')
    args_parser.add_argument('--device', '-d', type=str, help='Training device (cuda | cpu)',
                            default='cuda')
    args = args_parser.parse_args()

    init_training(args.config_file, args.save_dir, args.model_name, args.device)


if __name__ == "__main__":
    main()

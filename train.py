import os
import cv2
import json
import random
import argparse
import numpy as np
from pathlib import Path

from eprocessing.dataload import ImageDataset
from modelbuild.denoiser import DivergentRestorer
from modelbuild.nafnet import NAFNet

from eprocessing.etransforms import Scale, RandCrop, AddAWGN
from etrain.trainer import NNTrainer
from etrain.logger import MetricsLogger
from etrain.saver import NNSaver
from emetrics.metrics import *


DECONV1 = {'kern_size': (),
         'max_iters': 100,
         'iso': True}
DECONV2 = {'kern_size': (),
         'max_iters': 100,
         'iso': True}


class WeightClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'lmbda'):
            w = module.lmbda.data
            w = w.clamp(1e-12, 5)
            module.lmbda.data = w
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(1e-12, 5)
            module.rho.data = w


def seed_everything(seed=42):
    random.seed(seed)
    cv2.setRNGSeed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.RandomState(seed=seed)
    torch.manual_seed(seed)


def init_training(config_file: str, min_std: int, max_std: int, save_dir: str, model_name: str, device: str,
                  model_ckpt: str = None):
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
    train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=train_cfg['train']['batch_size'])
    eval_loader = torch.utils.data.DataLoader(eval_dset, shuffle=True, batch_size=train_cfg['eval']['batch_size'])

    save_dir_path = os.getcwd() + f'/{save_dir}'
    net_saver = NNSaver(save_dir_path, model_name)

    if train_cfg['train']['ckpt'] is not None:
        entry_model = DivergentRestorer([2, 8, 32], 3,
                                        3, 86,
                                        86, 8,
                                        output_activation=torch.nn.Sigmoid(), admms=[DECONV1, DECONV2])
        checkpoint = torch.load(train_cfg['train']['ckpt'], weights_only=False, map_location=device)
        entry_model.load_state_dict(checkpoint['model_state_dict'])
        # Freeze all
        print('WITH FROZEN!!!!')
        for param in entry_model.parameters():
            param.requires_grad = False
    else:
        entry_model = None

    model = DivergentRestorer([2, 8, 32], 3,
                              3, 86,
                              86, 8,
                              output_activation=torch.nn.Sigmoid(), admms=[DECONV1, DECONV2])

    # model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
    #                enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    # clipper = WeightClipper()
    # model.apply(clipper)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), train_cfg['lr'], betas=(0.9, 0.9))

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=1e-7, T_max=400000)

    eval_metrics = [PSNRMetric(device), SCCMetric(device), SSIMMetric(device), MAELoss(device), UIQMetric(device)]
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

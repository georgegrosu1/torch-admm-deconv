import os
import cv2
import json
import random
import argparse
import warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings(
    "ignore", 
    message="Importing from timm.models.layers is deprecated, please import via timm.layers"
    )

from admmtor.eprocessing.dataload import ImageDataset
from admmtor.modelbuild.denoiser import DivergentRestorer, DivergentRestorerResid
from admmtor.modelbuild.admm_fusion import ADMMFusion
from admmtor.modelbuild.nafnet import NAFNet
from admmtor.modelbuild.dranet import make_dranet
from admmtor.modelbuild.swinir import SwinIR
from admmtor.modelbuild.anet import ANet

from admmtor.eprocessing.etransforms import (
    Scale, 
    RandCrop,
    Flip,
    AddAWGN
    )
from admmtor.etrain.trainer import NNTrainer
from admmtor.etrain.logger import MetricsLogger
from admmtor.etrain.saver import NNSaver
from admmtor.emetrics.metrics import *


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
    
    
loss_funcs = {
    'charbonnier': CharbonnierLoss,
    'ssim_color_lab_loss': SSIMLabColorLoss,
    'alt_color_lab_loss': AlternativeSSIMLabColorLoss,
    'cascade_resid_loss': CascadeResidLoss,
    'mse_loss': MSE
}


def init_training(config_file: str, min_std: int, max_std: int, save_dir: str, model_name: str, device: str,
                  model_ckpt: str = None):
    config_file_path = os.getcwd() + f'/{config_file}'
    with open(config_file_path, 'r') as f:
        train_cfg = json.load(f)

    # Prepare train & eval data loaders
    im_shape = tuple(train_cfg['im_shape'])
    transforms = [RandCrop(im_shape), Scale(), Flip()]
    if max_std > 0: transforms += [AddAWGN(std_range=(min_std, max_std), both=False)]
    train_dset = ImageDataset(Path(train_cfg['train']['x_path']), Path(train_cfg['train']['y_path']),
                              transforms=transforms)
    eval_dset = ImageDataset(Path(train_cfg['eval']['x_path']), Path(train_cfg['eval']['y_path']),
                             transforms=transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dset, shuffle=True, batch_size=train_cfg['train']['batch_size'], 
        pin_memory=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(
        eval_dset, shuffle=True, batch_size=train_cfg['eval']['batch_size'], 
        pin_memory=True, num_workers=4)

    save_dir_path = os.getcwd() + f'/{save_dir}'
    net_saver = NNSaver(save_dir_path, model_name)
    
    model = DivergentRestorer(**train_cfg['model_params'])
    
    # model = ANet(**train_cfg['model_params'])
    
    # model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
    #                enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    
    # model = make_dranet(train_cfg['model_params'])
    # model = SwinIR(**train_cfg['model_params'])

    if train_cfg['train']['ckpt'] is not None:
        # modeldenoiser = DivergentRestorer(**train_cfg['model_params'])
        print("!!!!! LOADING CKPT !!!!!!!")
        checkpoint = torch.load(train_cfg['train']['ckpt'], weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Freeze all
        # print('WITH FROZEN!!!!')
        # for param in entry_model.parameters():
        #     param.requires_grad = False

    # clipper = WeightClipper()
    # model.apply(clipper)
    # model = ADMMFusion(modeldenoiser, modelresid, freeze_denoiser=True, freeze_denoiser_resid=False)
    model = model.to(device)
    # 1. Correctly partition the parameters
    params_1d = [p for p in model.parameters() if p.requires_grad and p.dim() != 2]
    params_2d = [p for p in model.parameters() if p.requires_grad and p.dim() == 2]
    
    # 2. Instantiate both optimizers independently
    opt_adamw = torch.optim.AdamW(params_1d, train_cfg['lr'], betas=(0.9, 0.9), eps=1e-12, weight_decay=1e-5)
    opt_muon = torch.optim.Muon(params_2d, train_cfg['lr'], momentum=0.95, weight_decay=1e-5)

    lr_scheduler_adamw = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_adamw, T_0=150000, eta_min=1e-11)
    lr_scheduler_muon = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_muon, T_0=150000, eta_min=1e-11)

    eval_metrics = [PSNRMetric(device), SSIMMetric(device), SCCMetric(device), UIQMetric(device)]
    loss_func = loss_funcs[train_cfg['lossf']](device)

    metrics_logger = MetricsLogger(loss_func, eval_metrics)
    net_trainer = NNTrainer(loss_func, eval_metrics, net_saver, metrics_logger)

    net_trainer.run(model, [opt_adamw, opt_muon], train_cfg['epochs'], train_loader, eval_loader, lr_scheduler=[lr_scheduler_adamw, lr_scheduler_muon])


def main():
    seed_everything()

    args_parser = argparse.ArgumentParser(description='Training script for image restoration')
    args_parser.add_argument('--config_file', '-c', type=str, help='Path to train config file',
                             default=r'configs/admm_cfg.json')
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
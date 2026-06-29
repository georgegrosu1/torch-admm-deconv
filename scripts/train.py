import os
import cv2
import json
import random
import argparse
import warnings
import numpy as np
from pathlib import Path
from typing import Union
import torch
torch.set_flush_denormal(True)
warnings.filterwarnings(
    "ignore", 
    message="Importing from timm.models.layers is deprecated, please import via timm.layers"
)
warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal"
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
    
model_funcs = {
    'divergent_restorer': DivergentRestorer,
    'nafnet': NAFNet,
    'dranet': make_dranet,
    'swinir': SwinIR,
}
    
loss_funcs = {
    'charbonnier': CharbonnierLoss,
    'ssim_color_lab_loss': SSIMLabColorLoss,
    'alt_color_lab_loss': AlternativeSSIMLabColorLoss,
    'cascade_resid_loss': CascadeResidLoss,
    'mse_loss': MSE,
    'psnr_loss': PSNRLoss,
}

opt_funcs = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'muon': torch.optim.Muon,
}

lr_scheduler_funcs = {
    'cosine_annealing': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_annealing_wr': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'exponential_lr': torch.optim.lr_scheduler.ExponentialLR,
    'multi_step_lr': torch.optim.lr_scheduler.MultiStepLR,
}


def init_training_objects(cfg_dict: dict, device: str):
    model = model_funcs[cfg_dict['model_name']](**cfg_dict['model_params'])
    model.to(device)
    
    if len(cfg_dict['optimizer_params']) == 2:
        # If there are two optimizers, we assume the first is for non 2D params and the second is for 2D params
        params_1d = [p for p in model.parameters() if p.requires_grad and p.dim() != 2]
        params_2d = [p for p in model.parameters() if p.requires_grad and p.dim() == 2]
        
        optims = [
            opt_funcs[cfg_dict['optimizer_params'][0]['optimizer']](params_1d, **cfg_dict['optimizer_params'][0]['cfgs']),
            opt_funcs[cfg_dict['optimizer_params'][1]['optimizer']](params_2d, **cfg_dict['optimizer_params'][1]['cfgs'])
        ]
    else:
        optims = [opt_funcs[cfg_dict['optimizer_params'][0]['optimizer']](model.parameters(), **cfg_dict['optimizer_params'][0]['cfgs'])]
    
    lr_schedulers = [
        lr_scheduler_funcs[lr['scheduler']](optims[0], **lr['cfgs'])
        for lr in cfg_dict['lr_params']
    ]
    
    loss_func = loss_funcs[cfg_dict['lossf']](device)
    
    return model, optims, lr_schedulers, loss_func


def load_model_from_ckpt(
    model: torch.nn.Module, 
    optimizers: Union[list, torch.optim.Optimizer], 
    lr_schedulers: Union[list, object], # Object used here as PyTorch scheduler base classes vary
    ckpt_path: str, 
    device: str
):
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 1. Load Model State
    model.load_state_dict(checkpoint['model_state_dict'])

    # 2. Load Optimizer State(s)
    if isinstance(optimizers, list):
        for opt, state in zip(optimizers, checkpoint['optimizer_state_dict']):
            opt.load_state_dict(state)
    else:
        optimizers.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # 3. Load Scheduler State(s)
    if 'scheduler_state_dict' in checkpoint:
        if isinstance(lr_schedulers, list):
            for sched, state in zip(lr_schedulers, checkpoint['scheduler_state_dict']):
                sched.load_state_dict(state)
        else:
            lr_schedulers.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Successfully loaded scheduler states.")
    else:
        print("Warning: 'scheduler_state_dict' not found in this checkpoint. Schedulers will start from scratch.")

    # 4. Extract metadata for resuming training
    start_epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('loss', None)
    
    print(f"Resuming from Epoch {start_epoch} with Val Loss: {val_loss}")
    
    return model, optimizers, lr_schedulers


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
    
    # model = DivergentRestorer(**train_cfg['model_params'])
    # model.to(device)
    # model = ANet(**train_cfg['model_params'])
    
    # model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
    #                enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    
    # model = make_dranet(train_cfg['model_params'])
    # model = SwinIR(**train_cfg['model_params'])

    # params_1d = [p for p in model.parameters() if p.requires_grad and p.dim() != 2]
    # params_2d = [p for p in model.parameters() if p.requires_grad and p.dim() == 2]
    
    # opt_adamw = torch.optim.AdamW(params_1d, train_cfg['lr'], betas=(0.9, 0.9), eps=1e-12, weight_decay=1e-5)
    # opt_muon = torch.optim.Muon(params_2d, train_cfg['lr'], momentum=0.95, weight_decay=1e-5)

    # lr_scheduler_adamw = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_adamw, T_0=150000, eta_min=1e-11)
    # lr_scheduler_muon = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_muon, T_0=150000, eta_min=1e-11)
    
    # Group them into lists for cleaner passing
    # optimizers_list = [opt_adamw, opt_muon]
    # schedulers_list = [lr_scheduler_adamw, lr_scheduler_muon]
    
    model, optimizers_list, schedulers_list, loss_func = init_training_objects(train_cfg, device)

    # 2. THEN, if a checkpoint exists, load the states INTO the objects
    if train_cfg['train']['ckpt'] is not None:
        model, optimizers_list, schedulers_list = load_model_from_ckpt(
            model,
            optimizers_list,
            schedulers_list,
            train_cfg['train']['ckpt'],
            device
        )
    
    eval_metrics = [PSNRMetric(device), SSIMMetric(device), SCCMetric(device), UIQMetric(device)]
    loss_func = loss_funcs[train_cfg['lossf']](device)

    metrics_logger = MetricsLogger(loss_func, eval_metrics)
    net_trainer = NNTrainer(loss_func, eval_metrics, net_saver, metrics_logger)

    net_trainer.run(model, optimizers_list, train_cfg['epochs'], train_loader, eval_loader, lr_scheduler=schedulers_list)


def main():
    seed_everything()

    args_parser = argparse.ArgumentParser(description='Training script for image restoration')
    args_parser.add_argument('--config_file', '-c', type=str, help='Path to train config file',
                             default=r'configs/dranet_cfg.json')
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
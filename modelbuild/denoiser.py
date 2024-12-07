import torch
import torch.nn as nn

from modelbuild.deconver import Deconvs
from modelbuild.updownscale import UpDownScale
from modelbuild.autoencoder import Autoencoder
from modelbuild.blocks import UpDownBock


DECONV1 = {'kern_size': (),
        'max_iters': 80,
         'rho': 0.2,
        'iso': True}
DECONV2 = {'kern_size': (),
         'max_iters': 80,
         'rho': 0.02,
         'iso': False}
DECONV3 = {'kern_size': (),
         'max_iters': 80,
         'rho': 0.004,
         'iso': True}
DECONV4 = {'kern_size': (),
         'max_iters': 80,
         'lmbda': 0.04,
         'iso': False}

AUTOENC1 = {'in_channels': 3,
            'enc_out_channels': [32, 64, 64, 128],
            'dec_out_channels': [64, 32, 32, 16],
            'kernel_sizes': [5, 7, 9, 11],
            'activation': torch.nn.ReLU6(),
            'pool_size': 3}
AUTOENC2 = {'in_channels': 3,
            'enc_out_channels': [16, 32, 32, 64],
            'dec_out_channels': [64, 32, 32, 16],
            'kernel_sizes': [13, 15, 17, 19],
            'activation': torch.nn.ReLU6(),
            'pool_size': 5}

UPDOWN1 = {'in_channels': 3,
           'out_channels': [16, 32, 32, 64, 128, 16],
           'kernel_sizes': [3, 5, 7, 9, 11, 13],
           'activation': torch.nn.ReLU6()}
UPDOWN2 = {'in_channels': 3,
           'out_channels': [16, 32, 32, 64, 128, 16],
           'kernel_sizes': [13, 15, 17, 19, 21, 23],
           'activation': torch.nn.ReLU6()}

ALL_OUT_CHS = (AUTOENC1['dec_out_channels'][-1] + AUTOENC2['dec_out_channels'][-1] +
                        UPDOWN1['out_channels'][-1] + UPDOWN2['out_channels'][-1])
AUTOENC_OUT = {'in_channels': ALL_OUT_CHS,
            'enc_out_channels': [16, 32, 64, 16],
            'dec_out_channels': [16, 64, 32, 3],
            'kernel_sizes': [11, 13, 15, 17],
            'activation': torch.nn.ReLU6(),
            'pool_size': 5}


class Denoiser(nn.Module):
    def __init__(self) -> None:
        super(Denoiser, self).__init__()
        self._init_denoise()


    def _init_denoise(self):
        self.denoise1 = nn.Sequential(
            Deconvs([DECONV1]),
            Autoencoder(**AUTOENC1)
        )
        self.denoise2 = nn.Sequential(
            Deconvs([DECONV2]),
            UpDownScale(**UPDOWN1)
        )
        self.denoise3 = nn.Sequential(
            Deconvs([DECONV3]),
            Autoencoder(**AUTOENC2)
        )
        self.denoise4 = nn.Sequential(
            Deconvs([DECONV4]),
            UpDownScale(**UPDOWN2)
        )

        self.out_block = Autoencoder(**AUTOENC_OUT)


    def forward_denoisers(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.denoise1(x),
                          self.denoise2(x),
                          self.denoise3(x),
                          self.denoise4(x)], dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_block(self.forward_denoisers(x))
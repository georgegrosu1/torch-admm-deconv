from typing import Dict

from modelbuild.autoencoder import *
from modelbuild.deconver import Deconvs
from modelbuild.updownscale import UpDownScale


class Restorer(nn.Module):
    def __init__(self,
                 inc_channels: int,
                 autoencoder_args: Dict,
                 updownscale_args: Dict,
                 deconvs_args: List[Dict]):
        super(Restorer, self).__init__()

        self.autoencoder = Autoencoder(**autoencoder_args)
        self.updownscale = UpDownScale(**updownscale_args)
        self.deconvs = Deconvs(deconvs_args)

        last_block_in_ch = (autoencoder_args['dec_out_channels'][-1] + updownscale_args['out_channels'][-1] +
                            len(deconvs_args) * inc_channels)
        last_block_out_ch = autoencoder_args['in_channels']
        self.out_block = UpDownBock(last_block_in_ch, last_block_in_ch//2, last_block_out_ch, 7, nn.ReLU6())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deconv_out = self.deconvs(x)
        comb = torch.cat([self.autoencoder(deconv_out), deconv_out, self.updownscale(deconv_out)], dim=1)
        return self.out_block(comb)

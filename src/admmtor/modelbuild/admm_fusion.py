import torch
import torch.nn as nn

from admmtor.modelbuild.denoiser import (
    DivergentRestorer,
    DivergentRestorerResid
)


class ADMMFusion(nn.Module):
    def __init__(self, 
                 denoiser: DivergentRestorer, 
                 denoiser_resid: DivergentRestorerResid,
                 freeze_denoiser: bool = True,
                 freeze_denoiser_resid: bool = False):
        super(ADMMFusion, self).__init__()
        self.denoiser = denoiser
        self.denoiser_resid = denoiser_resid
        if freeze_denoiser: self.freeze(self.denoiser)
        if freeze_denoiser_resid: self.freeze(self.denoiser_resid)
        
    def freeze(self, nn_module: nn.Module):
        for param in nn_module.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_denoised = self.denoiser(x)
        out_resid = self.denoiser_resid(x)
        return out_denoised, out_resid
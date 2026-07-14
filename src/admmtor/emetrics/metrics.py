from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                MultiScaleStructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio,
                                UniversalImageQualityIndex,
                                SpatialCorrelationCoefficient)
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from torchmetrics.regression import MeanSquaredError
from kornia.color import rgb_to_lab as kornia_rgb_to_lab


class Metric(ABC):
    m_name: str

    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        raise NotImplementedError


class MSE(Metric):
    m_name: str = 'mse'

    def __init__(self, device: str):
        super().__init__(device)
        self._func = MeanSquaredError().to(device)

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        return self._func(y_pred, y_true)


class SSIMLoss(Metric):
    m_name = 'ssim_loss'

    def __init__(self, device: str, data_range=1.0, kern_size: int = 11):
        super().__init__(device)
        self._func = StructuralSimilarityIndexMeasure(data_range=data_range, kernel_size=kern_size, k1=0.005, k2=0.025).to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return 1.0 - self._func(y_pred, y_true)
    

class DISTSLoss(Metric):
    m_name = 'dists_loss'

    def __init__(self, device: str):
        super().__init__(device)
        self._func = DeepImageStructureAndTextureSimilarity().to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self._func(y_pred, y_true)


class MAELoss(Metric):
    m_name = 'mae_loss'
    def __init__(self, device: str):
        super().__init__(device)
        self._func = torch.nn.L1Loss().to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self._func(y_pred, y_true)


class MSSSIMLoss(Metric):
    m_name = 'mssssim_loss'

    def __init__(self, device: str, data_range=1.0):
        super().__init__(device)
        self._func = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return 1 - self._func(y_pred, y_true)


class SSIMMetric(Metric):
    m_name = 'ssim'

    def __init__(self, device: str, data_range=1.0):
        super().__init__(device)
        self._func = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self._func(y_pred, y_true)
    
    
class DISTSMetric(Metric):
    m_name = 'dists'

    def __init__(self, device: str):
        super().__init__(device)
        self._func = DeepImageStructureAndTextureSimilarity().to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self._func(y_pred, y_true)


class MSSSIMMetric(Metric):
    m_name = 'msssim'

    def __init__(self, device: str, data_range=1.0):
        super().__init__(device)
        self._func = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self._func(y_pred, y_true)


class PSNRMetric(Metric):
    m_name = 'psnr'

    def __init__(self, device: str, data_range=1.0):
        super().__init__(device)
        self._func = PeakSignalNoiseRatio(data_range=data_range).to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self._func(y_pred, y_true)


class UIQMetric(Metric):
    m_name = 'uiq'

    def __init__(self, device: str):
        super().__init__(device)
        self._func = UniversalImageQualityIndex().to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self._func(y_pred, y_true)


class SCCMetric(Metric):
    m_name = 'scc'

    def __init__(self, device: str):
        super().__init__(device)
        self._func = SpatialCorrelationCoefficient().to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self._func(y_pred, y_true)


class PSNRLoss(Metric):
    m_name = 'psnr_loss'

    def __init__(self, device: str='cuda'):
        super(PSNRLoss, self).__init__(device)
        self.loss_weight = 1.0
        self.scale = torch.tensor([10]).to(device) / torch.log(torch.tensor([10]).to(device))
        self.to_y = False
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1).to(device)
        self.first = True

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert len(y_pred.size()) == 4
        pred, target = y_pred, y_true
        if self.to_y:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
    
    
class CharbonnierLoss(Metric):
    """Charbonnier Loss (L1)"""
    m_name = 'charbonnier_loss'

    def __init__(self, device: str='cuda', eps=1e-3):
        super(CharbonnierLoss, self).__init__(device)
        self.device = device
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.forward(y_pred, y_true)


class EdgeLoss(nn.Module):
    m_name = 'edge_loss'

    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)    # filter
        down = filtered[:, :, ::2, ::2]               # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4                  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.forward(y_pred, y_true)


class EdgeCharb(Metric):
    m_name = 'edge_carb_loss'

    def __init__(self, device: str):
        super().__init__(device)
        self._func1 = EdgeLoss().to(device)
        self._func2 = CharbonnierLoss().to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self._func1(y_pred, y_true) + 0.1 * self._func2(y_pred, y_true)
    
    
class FrequencyLoss(Metric):
    m_name = 'fft_loss'
    
    def __init__(self, device: str='cuda'):
        super(FrequencyLoss, self).__init__(device)
        self.loss_fn = nn.L1Loss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Compute 2D Real FFT
        fft_x = torch.fft.rfft2(y_pred, norm='ortho')
        fft_y = torch.fft.rfft2(y_true, norm='ortho')
        
        # Penalize differences in amplitude (magnitude)
        mag_x = torch.abs(fft_x)
        mag_y = torch.abs(fft_y)
        
        return self.loss_fn(mag_x, mag_y)
    
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.forward(y_pred, y_true)
    
    
class SSIMDISTSLoss(Metric):
    m_name = 'ssim_dists_loss'

    def __init__(self, device: str='cuda', ssim_weight=0.8, dists_weight=0.2):
        super(SSIMDISTSLoss, self).__init__(device)
        self.ssim_weight = ssim_weight
        self.dists_weight = dists_weight
        self.ssim_loss = SSIMLoss(device=device)
        self.dists_loss = DISTSLoss(device=device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        ssim_loss_val = self.ssim_loss(y_pred, y_true)
        dists_loss_val = self.dists_loss(y_pred, y_true)

        total_loss = (self.ssim_weight * ssim_loss_val) + (self.dists_weight * dists_loss_val)
        return total_loss
    

class MSSSIMDISTSLoss(Metric):
    m_name = 'msssim_dists_loss'

    def __init__(self, device: str='cuda', msssim_weight=0.8, dists_weight=0.2):
        super(MSSSIMDISTSLoss, self).__init__(device)
        self.msssim_weight = msssim_weight
        self.dists_weight = dists_weight
        self.msssim_loss = MSSSIMLoss(device=device)
        self.dists_loss = DISTSLoss(device=device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        msssim_loss_val = self.msssim_loss(y_pred, y_true)
        dists_loss_val = self.dists_loss(y_pred, y_true)

        total_loss = (self.msssim_weight * msssim_loss_val) + (self.dists_weight * dists_loss_val)
        return total_loss


class SSIMLabColorLoss(Metric):
    m_name = 'color_lab_loss'

    def __init__(self, device: str='cuda', ssim_weight=1.3, color_weight_ab=0.9, color_weight_l=0.3, reduction='mean'):
        super(SSIMLabColorLoss, self).__init__(device)
        self.ssim_weight = ssim_weight
        self.color_weight_ab = color_weight_ab
        self.color_weight_l = color_weight_l
        self.l1_loss = torch.nn.L1Loss(reduction=reduction)

        # SSIM Loss (assuming you have an SSIM implementation)
        # For demonstration, we'll use a placeholder for SSIM.
        # You'll need to replace this with your actual SSIM loss function.
        # Example: from pytorch_msssim import SSIM
        # self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)
        self.ssim_loss = SSIMLoss(device=device)

        # Kornia's rgb_to_lab expects input in [0, 1] for RGB.
        # Output L is [0, 100], a* and b* are [-100, 100] typically.
        self.rgb_to_lab_converter = kornia_rgb_to_lab
        # Or if you want to use the manual implementation (less recommended for production):
        # self.rgb_to_lab_converter = rgb_to_lab_manual

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
                Calculates the composite loss.

                Args:
                    denoised_image (torch.Tensor): The output of your denoising network (in RGB, range [0, 1]).
                                                   Shape: (B, 3, H, W)
                    reference_image (torch.Tensor): The ground truth/reference image (in RGB, range [0, 1]).
                                                    Shape: (B, 3, H, W)

                Returns:
                    torch.Tensor: The calculated composite loss.
                """
        # 1. SSIM Loss (on RGB)
        ssim_loss_val = self.ssim_loss(y_pred, y_true)  # This is (1 - SSIM)

        # 2. Convert to L*a*b*
        denoised_lab = self.rgb_to_lab_converter(y_pred)
        reference_lab = self.rgb_to_lab_converter(y_true)

        # Separate L, a*, b* channels
        denoised_L, denoised_a, denoised_b = denoised_lab[:, 0, :, :], denoised_lab[:, 1, :, :], denoised_lab[:, 2, :,
                                                                                                 :]
        reference_L, reference_a, reference_b = reference_lab[:, 0, :, :], reference_lab[:, 1, :, :], reference_lab[:,
                                                                                                      2, :, :]

        # Calculate L1 loss for each component
        # L* range is [0, 100], so normalize by 100
        loss_L = self.l1_loss(denoised_L, reference_L) / 100.0 if self.color_weight_l > 0 else 0.0

        # a* and b* range is approx [-100, 100], so normalize by 200
        loss_a = self.l1_loss(denoised_a, reference_a) / 200.0
        loss_b = self.l1_loss(denoised_b, reference_b) / 200.0

        color_loss_ab = (loss_a + loss_b) / 2

        total_loss = (self.ssim_weight * ssim_loss_val +
                      self.color_weight_ab * color_loss_ab +
                      self.color_weight_l * loss_L)

        return total_loss
    
    
class AlternativeSSIMLabColorLoss(Metric):
    m_name = 'alt_color_lab_loss'

    def __init__(self, device: str='cuda', ssim_weight=1.0, edge_weight=0.1, color_weight=1.0):
        super(AlternativeSSIMLabColorLoss, self).__init__(device)
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight

        # Base structural losses
        self.ssim_loss = SSIMLoss(device=device) 
        self.edge_loss = EdgeLoss()
        
        # Color loss
        self.charbonnier_loss = CharbonnierLoss()

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        ssim_loss_val = self.ssim_loss(y_pred, y_true)
        edge_loss_val = self.edge_loss(y_pred, y_true)
        color_loss_val = self.charbonnier_loss(y_pred, y_true)

        # 5. Composite Total
        total_loss = (self.ssim_weight * ssim_loss_val) + \
                     (self.edge_weight * edge_loss_val) + \
                     (self.color_weight * color_loss_val)
                     
        return total_loss
    
    
class CascadeResidLoss(Metric):
    m_name = 'cascade_resid_loss'
    
    def __init__(self, device: str='cuda'):
        super(CascadeResidLoss, self).__init__(device)
        self.loss = SSIMLabColorLoss(device=device)
        
    def __call__(self, y_pred: tuple[torch.Tensor, ...], y_true: torch.Tensor):
        out_denoised = y_pred[0]
        out_resid = y_pred[1]
        resid_true = y_true - out_denoised
        loss_resid = self.loss(out_resid, resid_true)
        
        return loss_resid
    
    
class ADMMFusionSSIM(Metric):
    m_name = 'admm_fusion_ssim'
    
    def __init__(self, device: str='cuda'):
        super(ADMMFusionSSIM, self).__init__(device)
        self.ssim_metric = SSIMMetric(device=device)
        
    def __call__(self, y_pred: tuple[torch.Tensor, ...], y_true: torch.Tensor):
        out_denoised = y_pred[0]
        out_resid = y_pred[1]
        fusion = out_denoised + out_resid
        ssim_val = self.ssim_metric(fusion, y_true)
        return ssim_val
    
    
class ADMMFusionPSNR(Metric):
    m_name = 'admm_fusion_psnr'
    
    def __init__(self, device: str='cuda'):
        super(ADMMFusionPSNR, self).__init__(device)
        self.psnr_metric = PSNRMetric(device=device)
        
    def __call__(self, y_pred: tuple[torch.Tensor, ...], y_true: torch.Tensor):
        out_denoised = y_pred[0]
        out_resid = y_pred[1]
        fusion = out_denoised + out_resid
        psnr_val = self.psnr_metric(fusion, y_true)
        return psnr_val
        

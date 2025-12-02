from abc import ABC

import torch
from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                MultiScaleStructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio,
                                UniversalImageQualityIndex,
                                SpatialCorrelationCoefficient)
from torchmetrics.regression import MeanSquaredError
from torchvision.transforms.functional import rgb_to_grayscale
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
        return self._func(y_true, y_pred)


class SSIMLoss(Metric):
    m_name = 'ssim_loss'

    def __init__(self, device: str, data_range=1.0, kern_size: int = 7):
        super().__init__(device)
        self._func = StructuralSimilarityIndexMeasure(data_range=data_range, kernel_size=kern_size).to(device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return 1.0 - self._func(y_pred, y_true)


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

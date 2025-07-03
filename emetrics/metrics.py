from abc import ABC

import torch
from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                MultiScaleStructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio,
                                UniversalImageQualityIndex,
                                SpatialCorrelationCoefficient)
from torchmetrics.regression import MeanSquaredError


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


class HSVLoss(Metric):
    m_name = 'hsv_loss'

    def __init__(self, device='cuda', v_weight=1.0, hs_weight=1.0, loss_fn=torch.nn.L1Loss()):
        super(HSVLoss, self).__init__(device)
        self.v_weight = v_weight
        self.hs_weight = hs_weight
        self.loss_fn = loss_fn

    @staticmethod
    def rgb_to_hsv(image_rgb: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of RGB images to HSV.

        Args:
            image_rgb (torch.Tensor): Input RGB image tensor.
                                      Shape: (..., 3, H, W) where ... is any number of batch dimensions.
                                      Values are expected to be in the range [0, 1].

        Returns:
            torch.Tensor: HSV image tensor.
                          Shape: (..., 3, H, W), with channels in the order H, S, V.
                          H, S, V values are in the range [0, 1].
        """
        if not isinstance(image_rgb, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor. Got {type(image_rgb)}")
        if image_rgb.shape[-3] != 3:
            raise ValueError(f"Input tensor must have 3 channels in the -3 dimension. Got shape {image_rgb.shape}")
        if image_rgb.min() < 0 or image_rgb.max() > 1:
            # This is a common expectation for neural network inputs.
            # You might want to normalize your image_rgb to [0, 1] before calling this function.
            print("Warning: Input RGB values are outside the [0, 1] range. Results might be unexpected.")

        # Get the original shape (excluding the channel dimension)
        original_shape = image_rgb.shape[:-3]
        h, w = image_rgb.shape[-2:]

        # Reshape to (N, 3, H, W) for easier processing, where N is product of batch dimensions
        # This handles any number of leading batch dimensions
        flat_rgb = image_rgb.view(-1, 3, h, w)

        # Split channels
        r, g, b = flat_rgb[:, 0, :, :], flat_rgb[:, 1, :, :], flat_rgb[:, 2, :, :]

        # Get max and min over RGB channels (Value and min_val)
        max_rgb, _ = torch.max(flat_rgb, dim=1)  # This will be V
        min_rgb, _ = torch.min(flat_rgb, dim=1)

        delta = max_rgb - min_rgb  # Chroma

        # Value (V)
        v = max_rgb

        # Saturation (S)
        # Avoid division by zero if max_rgb is 0 (black)
        s = torch.where(max_rgb != 0, delta / max_rgb, torch.zeros_like(delta))

        # Hue (H) - initialize with zeros
        h_channel = torch.zeros_like(delta)

        # Calculate Hue based on which channel is max
        # torch.where is crucial for differentiability here, avoiding Python if/else

        # Max is Red
        idx_r = (max_rgb == r) & (delta != 0)
        h_channel[idx_r] = ((g[idx_r] - b[idx_r]) / delta[idx_r]) % torch.tensor([6], device=image_rgb.device)

        # Max is Green
        idx_g = (max_rgb == g) & (delta != 0)
        h_channel[idx_g] = ((b[idx_g] - r[idx_g]) / delta[idx_g]) + torch.tensor([2], device=image_rgb.device)

        # Max is Blue
        idx_b = (max_rgb == b) & (delta != 0)
        h_channel[idx_b] = ((r[idx_b] - g[idx_b]) / delta[idx_b]) + torch.tensor([4], device=image_rgb.device)

        # Normalize H to [0, 1] by dividing by 6
        h_channel = h_channel / torch.tensor([6.0], device=image_rgb.device)

        # Stack H, S, V channels into a (N, 3, H, W) tensor
        hsv_image = torch.stack([h_channel, s, v], dim=1)

        # Reshape back to original batch dimensions
        final_shape = original_shape + (3, h, w)
        hsv_image = hsv_image.view(final_shape)

        return hsv_image

    def __call__(self, pred_rgb, target_rgb):
        """
        Computes the weighted HSV loss.
        Args:
            pred_rgb (torch.Tensor): Predicted/denoised image in RGB, shape (B, 3, H, W).
            target_rgb (torch.Tensor): Ground truth image in RGB, shape (B, 3, H, W).
        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        pred_hsv = self.rgb_to_hsv(pred_rgb)
        target_hsv = self.rgb_to_hsv(target_rgb)

        # Extract channels
        pred_h = pred_hsv[:, 0:1, :, :]  # Hue
        pred_s = pred_hsv[:, 1:2, :, :]  # Saturation
        pred_v = pred_hsv[:, 2:3, :, :]  # Value

        target_h = target_hsv[:, 0:1, :, :]
        target_s = target_hsv[:, 1:2, :, :]
        target_v = target_hsv[:, 2:3, :, :]

        # Calculate loss for each component
        loss_v = self.loss_fn(pred_v, target_v)
        loss_h = self.loss_fn(pred_h, target_h)
        loss_s = self.loss_fn(pred_s, target_s)

        # Combine Hue and Saturation loss for a single chrominance term
        loss_hs = loss_h + loss_s

        # Weighted sum of losses
        total_loss = self.v_weight * loss_v + self.hs_weight * loss_hs
        return total_loss


class SSIMHSVLoss(Metric):
    m_name = 'hybrid_ssim_hsv_loss'

    def __init__(self, device: str, lmbd_coeff: float = 0.7, data_range: float=1.0, kernel: int=7):
        super(SSIMHSVLoss, self).__init__(device)
        self._ssim_func = SSIMLoss(device, data_range=data_range, kern_size=kernel)
        self._hsv_func = HSVLoss(device)
        self.lmbd_coeff = lmbd_coeff

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self._ssim_func(y_pred, y_true) + self.lmbd_coeff * self._hsv_func(y_pred, y_true)

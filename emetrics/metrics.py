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

    def __init__(self, device: str, data_range=1.0):
        super().__init__(device)
        self._func = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

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

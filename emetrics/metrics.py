from abc import ABC, abstractmethod
from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                MultiScaleStructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio,
                                UniversalImageQualityIndex,
                                SpatialCorrelationCoefficient)


class Metric(ABC):
    m_name: str

    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        raise NotImplementedError


class SSIMLoss(Metric):
    m_name = 'ssim_loss'

    def __init__(self, device: str, data_range=1.0):
        super().__init__()
        self._func = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def __call__(self, y_true, y_pred):
        return 1.0 - self._func(y_true, y_pred)


class MSSSIMLoss(Metric):
    m_name = 'mssssim_loss'

    def __init__(self, device: str, data_range=1.0):
        super().__init__()
        self._func = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def __call__(self, y_true, y_pred):
        return 1 - self._func(y_true, y_pred)


class SSIMMetric(Metric):
    m_name = 'ssim'

    def __init__(self, device: str, data_range=1.0):
        super().__init__()
        self._func = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def __call__(self, y_true, y_pred):
        return self._func(y_true, y_pred)


class MSSSIMMetric(Metric):
    m_name = 'msssim'

    def __init__(self, device: str, data_range=1.0):
        super().__init__()
        self._func = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def __call__(self, y_true, y_pred):
        return self._func(y_true, y_pred)


class PSNRMetric(Metric):
    m_name = 'psnr'

    def __init__(self, device: str, data_range=1.0):
        super().__init__()
        self._func = PeakSignalNoiseRatio(data_range=data_range).to(device)

    def __call__(self, y_true, y_pred):
        return self._func(y_true, y_pred)


class UIQMetric(Metric):
    m_name = 'uiq'

    def __init__(self, device: str):
        super().__init__()
        self._func = UniversalImageQualityIndex().to(device)

    def __call__(self, y_true, y_pred):
        return self._func(y_true, y_pred)


class SCCMetric(Metric):
    m_name = 'scc'

    def __init__(self, device: str):
        super().__init__()
        self._func = SpatialCorrelationCoefficient().to(device)

    def __call__(self, y_true, y_pred):
        return self._func(y_true, y_pred)

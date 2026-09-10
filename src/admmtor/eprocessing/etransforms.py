import math

import torch
from torchvision.transforms.functional import crop, rotate
from torchvision.transforms import RandomRotation

class RandCrop(object):
    def __init__(self, im_shape):
        assert isinstance(im_shape, (int, tuple))
        if isinstance(im_shape, int):
            self.im_shape = (im_shape, im_shape)
        else:
            assert len(im_shape) == 2
            self.im_shape = im_shape

    def __call__(self, x_img, y_img):
        x_im, y_im = x_img, y_img

        _, h, w = y_im.shape
        new_h, new_w = self.im_shape

        top = torch.randint(0, h - new_h + 1, (1,)).tolist()[0]
        left = torch.randint(0, w - new_w + 1, (1,)).tolist()[0]

        x_im = crop(x_im, top, left, new_h, new_w)
        y_im = crop(y_im, top, left, new_h, new_w)

        return x_im, y_im


class Scale(object):
    def __call__(self, x_img, y_img) -> tuple[torch.Tensor, torch.Tensor]:
        return x_img / 255.0, y_img / 255.0
    
    
class Rotate(object):
    def __init__(self, degrees: int = 30):
        self.rotation = RandomRotation(degrees)
        self.degrees = degrees
    
    def __call__(self, x_img, y_img) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotation(x_img), self.rotation(y_img)
    
    
class Flip(object):
    def __call__(self, x_img, y_img) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < 0.5:
            x_img = torch.flip(x_img, [-1])
            y_img = torch.flip(y_img, [-1])
        if torch.rand(1) < 0.5:
            x_img = torch.flip(x_img, [-2])
            y_img = torch.flip(y_img, [-2])
        return x_img, y_img


class AddAWGN(object):
    def __init__(self,
                 mean: float = 0.0,
                 std_range: tuple[int, int] = (1, 1),
                 minval: float = 0.0,
                 maxval: float = 1.0,
                 both: bool = False):
        self.mean = mean
        self.std_range = std_range
        self.minval = minval
        self.maxval = maxval
        self.both = both


    def __call__(self, x_img, y_img) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.randint(self.std_range[0], self.std_range[1], (1,)).item() / 255.0
        awgn = torch.randn(x_img.shape).to(x_img.device) * std + self.mean
        if self.both:
            return torch.clamp(x_img + awgn, self.minval, self.maxval), torch.clamp(y_img + awgn, self.minval, self.maxval)
        return torch.clamp(x_img + awgn, self.minval, self.maxval), y_img


class AddRealisticSensorNoise(object):
    """Simulate a raw camera acquisition on images in the ``[minval, maxval]`` range.

    The model combines signal-dependent photon shot noise with dark current,
    pixel-response non-uniformity, row-correlated read noise, Gaussian read
    noise, and optional ADC quantization. Noise parameters are sampled once
    per transform call, making it suitable for noise-level augmentation.

    Args:
        full_well_range: Sensor capacity in electrons. Select the camera's
            full-well range; larger values give less relative shot noise.
        read_noise_std_range: Gaussian read-noise standard deviation expressed
            in 8-bit intensity units (0--255), matching the convention used by
            classic AWGN denoising benchmarks (for example, 15, 25, or 35).
            Internally this is converted to electrons via the sampled
            full-well capacity.
        dark_current_range: Dark electrons added to every pixel. Choose based
            on exposure temperature/time; use zero when negligible.
        prnu_std_range: Relative pixel-response standard deviation (for
            example, 0.01 means 1%). Use the measured PRNU or zero to disable.
        row_noise_std_range: Standard deviation of row-correlated noise in
            electrons. Use the measured row pattern level, or zero if absent.
        quantization_bits: ADC bit depth (1--24), typically the camera's
            value. Use ``None`` to disable quantization.
        minval: Minimum intensity represented by the input/output images.
        maxval: Maximum intensity represented by the input/output images.
        both: If true, apply the sampled sensor conditions to both images;
            otherwise only ``x_img`` is modified.
    """

    def __init__(
        self,
        full_well_range: tuple[float, float] = (1000.0, 10000.0),
        read_noise_std_range: tuple[float, float] = (15.0, 15.0),
        dark_current_range: tuple[float, float] = (0.0, 5.0),
        prnu_std_range: tuple[float, float] = (0.0, 0.02),
        row_noise_std_range: tuple[float, float] = (0.0, 1.0),
        quantization_bits: int | None = 12,
        minval: float = 0.0,
        maxval: float = 1.0,
        both: bool = False,
    ):
        self.full_well_range = self._validate_range(
            "full_well_range", full_well_range, minimum=0.0, strict_minimum=True
        )
        self.read_noise_std_range = self._validate_range(
            "read_noise_std_range", read_noise_std_range
        )
        self.dark_current_range = self._validate_range(
            "dark_current_range", dark_current_range
        )
        self.prnu_std_range = self._validate_range("prnu_std_range", prnu_std_range)
        self.row_noise_std_range = self._validate_range(
            "row_noise_std_range", row_noise_std_range
        )

        if quantization_bits is not None and (
            isinstance(quantization_bits, bool)
            or not isinstance(quantization_bits, int)
            or not 1 <= quantization_bits <= 24
        ):
            raise ValueError("quantization_bits must be None or an integer between 1 and 24")
        if not math.isfinite(minval) or not math.isfinite(maxval) or minval >= maxval:
            raise ValueError("minval must be finite and less than maxval")

        self.quantization_bits = quantization_bits
        self.minval = minval
        self.maxval = maxval
        self.both = both

    @staticmethod
    def _validate_range(
        name: str,
        value: tuple[float, float],
        minimum: float = 0.0,
        strict_minimum: bool = False,
    ) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values")

        lower, upper = float(value[0]), float(value[1])
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower > upper
            or lower < minimum
            or (strict_minimum and lower <= minimum)
        ):
            comparison = "greater than" if strict_minimum else "greater than or equal to"
            raise ValueError(
                f"{name} must be finite, ordered, and {comparison} {minimum}"
            )
        return lower, upper

    @staticmethod
    def _sample_uniform(value_range: tuple[float, float], device: torch.device) -> float:
        lower, upper = value_range
        if lower == upper:
            return lower
        return torch.empty((), device=device).uniform_(lower, upper).item()

    def _sample_noise_parameters(self, device: torch.device) -> tuple[float, float, float, float, float]:
        return (
            self._sample_uniform(self.full_well_range, device),
            self._sample_uniform(self.read_noise_std_range, device),
            self._sample_uniform(self.dark_current_range, device),
            self._sample_uniform(self.prnu_std_range, device),
            self._sample_uniform(self.row_noise_std_range, device),
        )

    def _add_noise(
        self,
        image: torch.Tensor,
        noise_parameters: tuple[float, float, float, float, float],
    ) -> torch.Tensor:
        if not image.is_floating_point():
            raise TypeError("AddRealisticSensorNoise requires floating-point image tensors")

        working_dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
        normalized = (
            (image.to(working_dtype) - self.minval) / (self.maxval - self.minval)
        ).clamp(0.0, 1.0)

        full_well, read_noise_std, dark_current, prnu_std, row_noise_std = noise_parameters

        if prnu_std > 0.0:
            pixel_response = (1.0 + torch.randn_like(normalized) * prnu_std).clamp_min(
                0.0
            )
            expected_electrons = normalized * pixel_response * full_well
        else:
            expected_electrons = normalized * full_well
        expected_electrons = expected_electrons + dark_current
        electrons = torch.poisson(expected_electrons.clamp_min(0.0))

        if read_noise_std > 0.0:
            read_noise_std_electrons = (read_noise_std / 255.0) * full_well
            electrons = electrons + torch.randn_like(electrons) * read_noise_std_electrons
        if row_noise_std > 0.0:
            row_noise_shape = (*electrons.shape[:-1], 1)
            electrons = electrons + (
                torch.randn(row_noise_shape, device=image.device, dtype=working_dtype)
                * row_noise_std
            )

        noisy_normalized = (electrons / full_well).clamp(0.0, 1.0)
        if self.quantization_bits is not None:
            quantization_levels = (1 << self.quantization_bits) - 1
            noisy_normalized = (
                torch.round(noisy_normalized * quantization_levels)
                / quantization_levels
            )

        return (
            noisy_normalized * (self.maxval - self.minval) + self.minval
        ).to(image.dtype)

    def __call__(
        self, x_img: torch.Tensor, y_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise_parameters = self._sample_noise_parameters(x_img.device)
        x_noisy = self._add_noise(x_img, noise_parameters)
        if self.both:
            return x_noisy, self._add_noise(y_img, noise_parameters)
        return x_noisy, y_img

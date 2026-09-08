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
    
    
class AddPoisson(object):
    def __init__(self,
                 lam_range: tuple[int, int] = (1, 1),
                 minval: float = 0.0,
                 maxval: float = 1.0,
                 both: bool = False):
        self.lam_range = lam_range
        self.minval = minval
        self.maxval = maxval
        self.both = both

    def __call__(self, x_img, y_img) -> tuple[torch.Tensor, torch.Tensor]:
        lam = torch.randint(self.lam_range[0], self.lam_range[1], (1,)).item() / 255.0
        poisson_noise = torch.poisson(torch.ones(x_img.shape) * lam).to(x_img.device)
        if self.both:
            return torch.clamp(x_img + poisson_noise, self.minval, self.maxval), torch.clamp(y_img + poisson_noise, self.minval, self.maxval)
        return torch.clamp(x_img + poisson_noise, self.minval, self.maxval), y_img
    
    
class InversePoisson(object):
    def __init__(self,
                 lam_range: tuple[int, int] = (1, 1),
                 minval: float = 0.0,
                 maxval: float = 1.0,
                 both: bool = False):
        self.lam_range = lam_range
        self.minval = minval
        self.maxval = maxval
        self.both = both

    def __call__(self, x_img, y_img) -> tuple[torch.Tensor, torch.Tensor]:
        lam = torch.randint(self.lam_range[0], self.lam_range[1], (1,)).item() / 255.0
        inv_poisson_noise = torch.poisson(torch.ones(x_img.shape) * lam).to(x_img.device)
        if self.both:
            return torch.clamp(x_img - inv_poisson_noise, self.minval, self.maxval), torch.clamp(y_img - inv_poisson_noise, self.minval, self.maxval)
        return torch.clamp(x_img - inv_poisson_noise, self.minval, self.maxval), y_img

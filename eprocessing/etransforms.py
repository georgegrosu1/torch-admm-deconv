import torch
from torchvision.transforms.functional import crop
from typing import Tuple

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

        h, w = y_im.shape[-2:]
        new_h, new_w = self.im_shape

        top = torch.randint(0, h - new_h + 1, (1,)).tolist()[0]
        left = torch.randint(0, w - new_w + 1, (1,)).tolist()[0]

        x_im = crop(x_im, top, left, new_h, new_w)
        y_im = crop(y_im, top, left, new_h, new_w)

        return x_im, y_im


class Scale(object):
    def __call__(self, x_img, y_img):
        return x_img / 255, y_img / 255


class AddAWGN(object):
    def __init__(self,
                 mean: float = 0.0,
                 std_range: Tuple[int, int] = (1, 1),
                 minval: float = 0.0,
                 maxval: float = 1.0,
                 both: bool = False):
        self.mean = mean
        self.std_range = std_range
        self.minval = minval
        self.maxval = maxval
        self.both = both


    def __call__(self, x_img, y_img):
        std = torch.randint(self.std_range[0], self.std_range[1], (1,)).item() / 255.0
        awgn = torch.clamp(torch.randn(x_img.shape).to(x_img.device) * std + self.mean, self.minval, self.maxval)
        if self.both:
            return torch.clamp(x_img + awgn, 0.0, 1.0), torch.clamp(y_img + awgn, self.minval, self.maxval)
        return torch.clamp(x_img + awgn, self.minval, self.maxval)

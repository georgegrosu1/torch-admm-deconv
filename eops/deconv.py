import torch
import torch.nn.functional as F

from typing import Tuple


def torch_abs2(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x) ** 2


def hard_thresh(x: torch.Tensor, tau: float) -> torch.Tensor:
    return x * (torch.abs(x) > tau)


def soft_thresh(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(x) * torch.maximum(torch.abs(x)-tau, torch.tensor([0]))


def block_thresh(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return torch.maximum(1 - tau / pixelnorm(x), torch.tensor([0], dtype=x.dtype, device=x.device)) * x


def pixelnorm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(x ** 2, (0, 1)))


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def conv_circular(x: torch.Tensor, w: torch.Tensor, pads: Tuple, groups: int) -> torch.Tensor:
    return F.conv2d(F.pad(x, pads, mode='circular'), w, groups=groups)


def fft_admm_tv(xin: torch.Tensor,
                lmbd: torch.Tensor,
                rho: torch.Tensor,
                kern: torch.Tensor,
                iso: bool=False,
                maxit: int=100) -> torch.Tensor:

    B, C, H_im, W_im = xin.shape

    tau = lmbd / rho

    if kern.numel() == 0:
        sigma = torch.tensor([1], dtype=xin.dtype, device=xin.device)
    else:
        sigma = torch.fft.rfftn(kern, s=(H_im,W_im), dim=(2,3))

    dx_base = torch.tensor([[[[0, 0], [-1, 1]]]], dtype=xin.dtype, device=xin.device)
    dy_base = torch.tensor([[[[0, -1], [0, 1]]]], dtype=xin.dtype, device=xin.device)

    delta_dx = torch.fft.rfftn(dx_base, s=(1,1,H_im,W_im))
    delta_dy = torch.fft.rfftn(dy_base, s=(1,1,H_im,W_im))

    freq_c = 1 / (torch_abs2(sigma) + rho * (torch_abs2(delta_dx) + torch_abs2(delta_dy)))

    thresh = block_thresh if iso else soft_thresh

    x = torch.zeros_like(xin, dtype=xin.dtype, device=xin.device)

    z_x = torch.zeros_like(xin, dtype=xin.dtype, device=xin.device)
    z_y = torch.zeros_like(xin, dtype=xin.dtype, device=xin.device)

    u_x = torch.zeros_like(xin, dtype=xin.dtype, device=xin.device)
    u_y = torch.zeros_like(xin, dtype=xin.dtype, device=xin.device)

    w_x_base = dx_base.repeat((C, 1, 1, 1))
    w_y_base = dy_base.repeat((C, 1, 1, 1))
    w_x_conj = dx_base.flip(2,3).repeat((C, 1, 1, 1))
    w_y_conj = dy_base.flip(2,3).repeat((C, 1, 1, 1))

    def Dx(a: torch.Tensor) -> torch.Tensor:
        return conv_circular(a, w_x_base, pads=(1, 0, 1, 0), groups=C)

    def Dy(a: torch.Tensor) -> torch.Tensor:
        return conv_circular(a, w_y_base, pads=(1, 0, 1, 0), groups=C)

    def Dx_t(a: torch.Tensor) -> torch.Tensor:
        return conv_circular(a, w_x_conj, pads=(0, 1, 0, 1), groups=C)

    def Dy_t(a: torch.Tensor) -> torch.Tensor:
        return conv_circular(a, w_y_conj, pads=(0, 1, 0, 1), groups=C)

    if kern.numel() == 0:
        H_t = identity
    else:
        k_kern_t = kern.flip((2,3)).repeat((C,1,1,1))
        padup = torch.ceil(torch.tensor(kern.size(2) - 1, dtype=xin.dtype, device=xin.device) / 2)
        paddown = torch.floor(torch.tensor(kern.size(2) - 1, dtype=xin.dtype, device=xin.device) / 2)
        padleft = torch.ceil(torch.tensor(kern.size(3) - 1, dtype=xin.dtype, device=xin.device) / 2)
        padright = torch.floor(torch.tensor(kern.size(3) - 1, dtype=xin.dtype, device=xin.device) / 2)

        pad1 = (int(padup), int(paddown), int(padleft), int(padright))
        pad2 = (int(paddown), int(padup), int(padright), int(padleft))

        def htran(x: torch.Tensor) -> torch.Tensor:
            return conv_circular(x, k_kern_t, pad2, groups=C)

        H_t = htran

    for _ in torch.arange(0, maxit):
        rfft_term = torch.fft.rfftn(H_t(xin) + rho * (Dx_t(z_x-u_x) + Dy_t(z_y-u_y)), dim=(2,3))

        x = torch.fft.irfftn(freq_c * rfft_term, (H_im, W_im), dim=(2,3))

        dx_k = Dx(x)
        dy_k = Dy(x)

        z_x = thresh(dx_k + u_x, tau)
        z_y = thresh(dy_k + u_y, tau)

        u_x += dx_k - z_x
        u_y += dy_k - z_y

    return x

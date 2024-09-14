import torch
import torch.nn.functional as F

from typing import Tuple


def torch_abs2(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.abs(x), 2)


def hard_thresh(x: torch.Tensor, tau: float) -> torch.Tensor:
    return x * (torch.abs(x) > tau)


def soft_thresh(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(x) * torch.maximum(torch.abs(x)-tau, torch.tensor([0]))


def block_thresh(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.maximum(1 - tau / pixelnorm(x), torch.tensor([0])) * x


def pixelnorm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.pow(x, 2), (2, 3)))


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

    # make x from (batch, ch, h, w) -> (h, w, batch, ch)
    xin = xin.permute(2,3,0,1)
    M, N, B, P = xin.shape

    tau = lmbd / rho

    if kern.numel() == 0:
        k_kern = kern
        sigma = torch.tensor([1])
    else:
        k_kern = torch.constant_pad_nd(kern, (0, M-kern.shape[0], 0, N-kern.shape[1]))
        sigma = torch.fft.rfftn(k_kern)

    dx_base = torch.tensor([[1, 0], [-1, 0]])
    dy_base = torch.tensor([[1, -1], [0, 0]])
    dx_filter = torch.constant_pad_nd(dx_base, (0, M - 2, 0, N - 2))
    dy_filter = torch.constant_pad_nd(dy_base, (0, M - 2, 0, N - 2))

    delta_dx = torch.fft.rfftn(dx_filter)
    delta_dy = torch.fft.rfftn(dy_filter)

    freq_c = 1 / (torch_abs2(sigma) + rho * (torch_abs2(delta_dx) + torch_abs2(delta_dy)))

    thresh = block_thresh if iso else soft_thresh

    x = torch.zeros((M, N, B, P))
    dx_k = torch.zeros((M, N, 2*B, P))
    z = torch.zeros((M, N, 2*B, P))
    u = torch.zeros((M, N, 2*B, P))

    w_normal_1 = torch.tensor([[1, -1], [0, 0]])
    w_normal_2 = torch.tensor([[1, 0], [-1, 0]])
    w_normal = torch.cat((w_normal_1[:,:,torch.newaxis,torch.newaxis],
                          w_normal_2[:,:,torch.newaxis,torch.newaxis]), 3)
    w_tr_1 = torch.tensor([[0, 0], [-1, 1]])
    w_tr_2 = torch.tensor([[0, -1], [0, 1]])
    w_tr = torch.cat((w_tr_1[:,:,torch.newaxis,torch.newaxis],
                      w_tr_2[:,:,torch.newaxis,torch.newaxis]), 2)

    if kern.numel() == 0:
        H = identity
        H_t = identity
    else:
        k_kern = k_kern.repeat((1,1,1,B))
        k_kern_t = k_kern.flip((0,1))
        padup = torch.ceil(torch.tensor(k_kern.size(0) - 1) / 2)
        paddown = torch.floor(torch.tensor(k_kern.size(0) - 1) / 2)
        padleft = torch.ceil(torch.tensor(k_kern.size(1) - 1) / 2)
        padright = torch.floor(torch.tensor(k_kern.size(1) - 1) / 2)

        pad1 = (padup, paddown, padleft, padright)
        pad2 = (paddown, padup, padright, padleft)

        def hnorm(x: torch.Tensor) -> torch.Tensor:
            return conv_circular(x, k_kern, pad1, groups=B)

        def htran(x: torch.Tensor) -> torch.Tensor:
            return conv_circular(x, k_kern_t, pad2, groups=B)

        H = hnorm
        H_t = htran

        for _ in torch.range(0, maxit):
            rfft_term = torch.fft.rfftn(H_t(xin) + rho * conv_circular(z-u, w_tr, (0,1,0,1), B), dim=(0,1))
            x = torch.fft.irfftn(freq_c * rfft_term, (0,1))
            dx_k = conv_circular(x, w_normal, (1,0,1,0), B)
            z = thresh(dx_k + u, tau)
            u += dx_k - z

        x = x.permute(2,3,0,1)

        return x

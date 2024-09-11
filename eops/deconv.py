import torch


def torch_abs2(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.abs(x), 2)


def hard_thresh(x: torch.Tensor, tau: float) -> torch.Tensor:
    return x * (torch.abs(x) > tau)


def soft_thresh(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(x) * torch.maximum(torch.abs(x)-tau, torch.tensor([0]))


def block_thresh(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.maximum(1 - tau / pixelnorm(x), torch.tensor([0])) * x


def pixelnorm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.pow(x, 2)), (2, 3))


def fft_admm_tv(xin: torch.Tensor,
                lmbd: float,
                rho: float,
                kern: torch.Tensor,
                iso: bool=False,
                maxit: int=100):

    xin = xin.permute(2,3,0,1) # make x from (batch, ch, h, w) -> (h, w, batch, ch)
    M, N, B, P = xin.shape

    tau = lmbd / rho

    if kern.numel() == 0:
        sigma = torch.tensor([1])
    else:
        k_kern = torch.constant_pad_nd(kern, (0, M-kern.shape[0], 0, N-kern.shape[1]))
        sigma = torch.fft.rfftn(k_kern)

    dx_filter = torch.tensor([[1, -1], [0, 0]])
    dy_filter = torch.tensor([[1, 0], [-1, 0]])
    dx_filter = torch.constant_pad_nd(dx_filter, (0, M - 2, 0, N - 2))
    dy_filter = torch.constant_pad_nd(dy_filter, (0, M - 2, 0, N - 2))

    delta_dx = torch.fft.rfftn(dx_filter)
    delta_dy = torch.fft.rfftn(dy_filter)

    freq_c = 1 / (torch_abs2(sigma) + rho * (torch_abs2(delta_dx) + torch_abs2(delta_dy)))

    thresh = block_thresh if iso else soft_thresh

    x = torch.zeros((M, N, B, P))
    dx_k = torch.zeros((M, N, 2*B, P))
    z = torch.zeros((M, N, 2*B, P))
    u = torch.zeros((M, N, 2*B, P))



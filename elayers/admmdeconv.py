import torch
from typing import Tuple, Callable
from eops.deconv import identity, fft_admm_tv


class ADMMDeconv(torch.nn.Module):
    def __init__(self,
                 kern_size: Tuple[int, int],
                 max_iters: int,
                 lmbda: float = None,
                 rho: float = None,
                 iso: bool = True,
                 bias: bool = False,
                 activation: Callable = identity):
        super(ADMMDeconv, self).__init__()

        self._init_kernel(kern_size)
        self.max_iters = max_iters
        self._init_lmbd(lmbda)
        self._init_rho(rho)
        self.iso = iso
        self._init_bias(bias)
        self.activation = activation


    def _init_rho(self, rho: float):
        if not rho:
            self.rho = torch.nn.Parameter(torch.empty(1,), requires_grad=True)
            torch.nn.init.uniform_(self.rho, a=0.0, b=1.0)
        else:
            rho = torch.tensor([rho], dtype=torch.float32)
            self.register_buffer('rho', rho)


    def _init_lmbd(self, lmbda: float):
        if not lmbda:
            self.lmbda = torch.nn.Parameter(torch.empty(1,), requires_grad=True)
            torch.nn.init.uniform_(self.lmbda, a=0.0, b=1.0)
        else:
            lmbda = torch.tensor([lmbda], dtype=torch.float32)
            self.register_buffer('lmbda', lmbda)


    def _init_kernel(self, kern_size):
        if kern_size:
            kshape = (1, 1, *kern_size)
            self.w = torch.nn.Parameter(torch.empty(kshape), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.w)
        else:
            w = torch.tensor([], dtype=torch.float32)
            self.register_buffer('w', w)


    def _init_bias(self, bias):
        if bias:
            self.b = torch.nn.Parameter(torch.empty(1,), requires_grad=True)
            torch.nn.init.uniform_(self.b, a=0.0, b=1.0)
        else:
            b = torch.tensor([0], dtype=torch.float32)
            self.register_buffer('b', b)


    def forward(self, x):
        return self.activation(fft_admm_tv(x, self.lmbda, self.rho, self.w, self.iso, self.max_iters) + self.b)

import torch
from typing import Tuple, Callable
from admmtor.eops.deconv import identity, fft_admm_tv


class ADMMDeconv(torch.nn.Module):
    def __init__(self,
                 kern_size: Tuple[int, int],
                 max_iters: int,
                 lmbda: float = None,
                 rho: float = None,
                 iso: bool = True,
                 bias: bool = False,
                 activation: Callable = identity,
                 per_iter: bool = False):
        super(ADMMDeconv, self).__init__()

        self._init_kernel(kern_size)
        self.max_iters = max_iters
        # per_iter: learn a separate lmbda / rho for each unrolled iteration (all start from the same value)
        self._n_params = max_iters if per_iter else 1
        self._init_lmbd(lmbda)
        self._init_rho(rho)
        self.iso = iso
        self._init_bias(bias)
        self.activation = activation


    def _init_rho(self, rho: float):
        if not rho:
            init = torch.nn.init.uniform_(torch.empty(1,), a=0.0, b=1.0)
            self.rho = torch.nn.Parameter(init.repeat(self._n_params), requires_grad=True)
        else:
            rho = torch.tensor([rho], dtype=torch.float32)
            self.register_buffer('rho', rho)


    def _init_lmbd(self, lmbda: float):
        if not lmbda:
            init = torch.nn.init.uniform_(torch.empty(1,), a=0.0, b=1.0)
            self.lmbda = torch.nn.Parameter(init.repeat(self._n_params), requires_grad=True)
        else:
            lmbda = torch.tensor([lmbda], dtype=torch.float32)
            self.register_buffer('lmbda', lmbda)


    def _init_kernel(self, kern_size: Tuple[int, int]):
        if kern_size:
            kshape = (1, 1, *kern_size)
            self.w = torch.nn.Parameter(torch.empty(kshape), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.w)
        else:
            w = torch.tensor([], dtype=torch.float32)
            self.register_buffer('w', w)


    def _init_bias(self, bias: bool):
        if bias:
            self.b = torch.nn.Parameter(torch.empty(1,), requires_grad=True)
            torch.nn.init.uniform_(self.b, a=0.0, b=1.0)
        else:
            b = torch.tensor([0], dtype=torch.float32)
            self.register_buffer('b', b)


    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints saved with a single lmbda / rho load into a per-iteration layer
        for name in ('lmbda', 'rho'):
            key = prefix + name
            if key in state_dict and state_dict[key].numel() == 1 and getattr(self, name).numel() > 1:
                state_dict[key] = state_dict[key].repeat(getattr(self, name).numel())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(fft_admm_tv(x, self.lmbda, self.rho, self.w, self.iso, self.max_iters) + self.b)

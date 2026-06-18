
import re
import torch
from torch import nn


@torch.no_grad()
def default_init_weights(
    nn_modules: nn.Module | list[nn.Module], 
    weights_att_names: list[str] = ['weight', 'bias'],
    init_func: callable = nn.init.kaiming_normal_,
    bias_eps: float = 1e-12
    ) -> None:
    nn_modules = nn_modules if isinstance(nn_modules, list) else [nn_modules]
    
    supported_types = re.compile(r'(?i)(?:conv|linear|norm|pool)')
    for nn_module in nn_modules:
        if supported_types.search(nn_module.__class__.__name__):
            for w_name in weights_att_names:
                if hasattr(nn_module, w_name):
                    param = getattr(nn_module, w_name)
                    if param is not None:
                        if 'bias' in w_name:
                            param.data.fill_(bias_eps)
                        else:
                            # Only apply fan-based initialization if tensor has at least 2 dimensions
                            if param.dim() >= 2:
                                init_func(param)
                            else:
                                # For 1D tensors (e.g., some biases), use normal initialization
                                nn.init.normal_(param, std=0.01)
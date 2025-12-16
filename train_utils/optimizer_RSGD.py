import torch
from torch.optim.optimizer import Optimizer
import random

episilon = 1e-8


def unit(v, dim: int = -1, eps: float = 1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm


def norm(v, dim: int = -1):
    assert len(v.size()) in [2, 3]
    return v.norm(p=2, dim=dim, keepdim=True)

def qr_retraction(tan_vec):
    """
    Batch QR retraction for Stiefel manifold.
    tan_vec: (B, p, n), p <= n
    returns: (B, p, n)
    """
    
    tan_vec_transposed = tan_vec.transpose(-1, -2)  # (B, n, p)
    q, r = torch.linalg.qr(tan_vec_transposed)      # q: (B, n, p)
    
    d = torch.diagonal(r, dim1=-2, dim2=-1)          # (B, p)
    ph = d.sign()                                    # (B, p)
    
    ph_expanded = ph.unsqueeze(-2)                   # (B, 1, p)
    q_corrected = q * ph_expanded
    
    return q_corrected.transpose(-1, -2) # (B, p, n)



class RSGDG(Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum: int = 0,
        dampening: int = 0,
        weight_decay: int = 0,
        nesterov: bool = False,
        stiefel: bool = False,
        omega: int = 0,
        grad_clip=None,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
            omega=0,
            grad_clip=grad_clip,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(RSGDG, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(RSGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            stiefel = group["stiefel"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if not p.dim() == 3:
                    raise NotImplementedError 
                
                X = p.data # (B, D, D)
                
                if stiefel:
                    # Euclidean G
                    G = p.grad.data.view(p.size()[0], p.size()[1], -1) # (B, p, n)

                    lr = group["lr"]

                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros_like(G)
                        if p.is_cuda:
                            param_state["momentum_buffer"] = param_state["momentum_buffer"].cuda()

                    V = param_state["momentum_buffer"] # V_{t-1}, (B, p, n)

                    # === Riemannian Gradient ===
                    # grad f(X) = G - X * sym(X^T G)
                    # X: (B, p, n), G: (B, p, n)

                    XT_G = torch.bmm(X.transpose(-1, -2), G) # (B, n, p)
                
                    sym_XT_G = 0.5 * (XT_G + XT_G.transpose(-1, -2)) # (B, n, p)

                    X_sym_XT_G = torch.bmm(X, sym_XT_G) # (B, p, n)

                    grad_X = G - X_sym_XT_G # (B, p, n)

                    V.mul_(momentum).sub_(grad_X, alpha=lr) # V_t = V_t * momentum - lr * grad_X

                    tan_vec_input = X + V # (B, p, n)
                    X_new = qr_retraction(tan_vec_input) # (B, p, n)
                    
                    p.data.copy_(X_new.view(p.size()))
                    

                else:
                    raise NotImplementedError("Stiefel=False (Euclidean SGD) is not implemented in this version.")

        return loss
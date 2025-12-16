# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is originally from: https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/master/stiefel_optimizer.py

import random

import torch
from torch.optim.optimizer import Optimizer


def unit(v, dim: int = -1, eps: float = 1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm


def norm(v, dim: int = -1):
    assert len(v.size()) in [2, 3]
    return v.norm(p=2, dim=dim, keepdim=True)

def matrix_norm_one_batch(W):
    """
    Compute per-batch 1-norm: max column sum for each batch.
    W: (B, p, n)
    returns: (B,)
    """
    # debug: check the shape of W ========================================
    # print(W.shape)
    # end debug ==========================================================
    
    col_sum = torch.sum(torch.abs(W), dim=-2)  # (B, n)
    # debug: check the shape of col_sum ====================================
    # print(col_sum.shape)
    # end debug ============================================================
    max_sum = torch.max(col_sum)
    return max_sum


def cayley_loop_batch(X, W, tan_vec, t):
    """
    Batch Cayley flow update.
    X, W, tan_vec: (B, p, n)
    t: scalar
    returns: (B, p, n)
    """
    Y = X + t * tan_vec
    for _ in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))
    return Y


# def qr_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
#     [num_block, p, n] = tan_vec.size()
#     tan_vec = tan_vec.transpose(-1, -2)
#     q, r = torch.linalg.qr(tan_vec)
#     d = torch.diag(r, 0)
#     ph = d.sign()
#     q *= ph.expand_as(q)
#     q.t_()

#     return q
def qr_retraction(tan_vec):
    """
    Batch QR retraction for Stiefel manifold.
    tan_vec: (B, p, n), p <= n
    returns: (B, p, n)
    """
    tan_vec = tan_vec.transpose(-1, -2)  # (B, n, p)
    q, r = torch.linalg.qr(tan_vec)      # q: (B, n, n), r: (B, n, p)
    d = torch.diagonal(r, dim1=-2, dim2=-1)  # (B, p)
    ph = d.sign().unsqueeze(-2)               # (B,1,p)
    q = q * ph
    q = q.transpose(-1, -2)                  # (B, p, n)
    return q

episilon = 1e-8


class SGDG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'stiefel'.

        If stiefel is True, the variables will be updated by SGD-G proposed
        as decorrelated weight matrix.

        If stiefel is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        stiefel (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case stiefel is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case stiefel is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

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
        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # print("!!!!!!!!!!!!!!!!!!!!!!!In optimizer.step()!!!!!!!!!!!")
        loss = None
        if closure is not None:
            loss = closure()
        
        # Debug: remove the optimizer to Check the loss change ==============
        # return loss
        # end debug ====================================================

        for group in self.param_groups:
            momentum = group["momentum"]
            stiefel = group["stiefel"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if not p.dim() == 3:
                    raise NotImplementedError
                    unity, _ = unit(p.data.view(p.size()[0], -1))
                else:
                    unity, _ = unit(p.data.view(p.size()[0], p.size()[1], -1))
                    
                if stiefel:
                    assert unity.dim() == 3
                    weight_decay = group["weight_decay"]
                    dampening = group["dampening"]
                    nesterov = group["nesterov"]

                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        unity = qr_retraction(unity)

                    g = p.grad.data.view(p.size()[0], p.size()[1], -1)

                    lr = group["lr"]

                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros(g.transpose(-1, -2).size())
                        if p.is_cuda:
                            param_state["momentum_buffer"] = param_state[
                                "momentum_buffer"
                            ].cuda()

                    V = param_state["momentum_buffer"]
                    V = momentum * V - g.transpose(-1, -2)
                    MX = torch.bmm(V, unity)
                    XMX = torch.bmm(unity, MX)
                    XXMX = torch.bmm(unity.transpose(-1, -2), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.transpose(-1, -2)
                    t = 0.5 * 2 / (matrix_norm_one_batch(W) + episilon)
                    # print(f"t: {t}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # print(f"lr: {lr}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    alpha = min(t, lr)
                    # exit(0)


                    p_new = cayley_loop_batch(unity.transpose(-1, -2), W, V, alpha)
                    V_new = torch.bmm(W, unity.transpose(-1, -2))  # n-by-p
                    #                     check_identity(p_new.t())
                    p.data.copy_(p_new.view(p.size()))
                    V.copy_(V_new)
                    
                    
                # unity, _ = unit(p.data.view(p.size()[0], -1))
                # if stiefel and unity.size()[0] <= unity.size()[1]:
                #     weight_decay = group["weight_decay"]
                #     dampening = group["dampening"]
                #     nesterov = group["nesterov"]

                #     rand_num = random.randint(1, 101)
                #     if rand_num == 1:
                #         unity = qr_retraction(unity)

                #     g = p.grad.data.view(p.size()[0], -1)

                #     lr = group["lr"]

                #     param_state = self.state[p]
                #     if "momentum_buffer" not in param_state:
                #         param_state["momentum_buffer"] = torch.zeros(g.t().size())
                #         if p.is_cuda:
                #             param_state["momentum_buffer"] = param_state[
                #                 "momentum_buffer"
                #             ].cuda()

                #     V = param_state["momentum_buffer"]
                #     V = momentum * V - g.t()
                #     MX = torch.mm(V, unity)
                #     XMX = torch.mm(unity, MX)
                #     XXMX = torch.mm(unity.t(), XMX)
                #     W_hat = MX - 0.5 * XXMX
                #     W = W_hat - W_hat.t()
                #     t = 0.5 * 2 / (matrix_norm_one(W) + episilon)
                #     alpha = min(t, lr)

                #     p_new = Cayley_loop(unity.t(), W, V, alpha)
                #     V_new = torch.mm(W, unity.t())  # n-by-p
                #     #                     check_identity(p_new.t())
                #     p.data.copy_(p_new.view(p.size()))
                #     V.copy_(V_new)

                else:
                    raise NotImplementedError
        return loss






def msign(x: torch.Tensor, steps=5, eps=1e-20):
    a, b, c, y = 3.4445, -4.7750, 2.0315, x.to(torch.bfloat16).clone()
    # We only target block-wise diagnal orthogonal matrix in this project
    assert y.dim() == 3 and y.shape[-2] == y.shape[-1]
    y /= ((y**2).sum(axis=[-2, -1], keepdims=True) + eps)**0.5
    for i in range(steps):
        y2 = torch.bmm(y, y.transpose(-1, -2))
        y4 = torch.bmm(y2, y2)
        if i == 0:
            n = (torch.sum(y4.pow(2), dim=[-2, -1], keepdim=True) + eps).pow(0.125)
            y, y2, y4 = y / n, y2 / n.pow(2), y4 / n.pow(4)
            
        term = b * y2 + c * y4
        m_term = torch.bmm(term, y)
        y = a * y + m_term
    return y


def skew(x: torch.Tensor):
    return (x - x.transpose(-1, -2)) / 2


class SGDGMuon(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'stiefel'.

        If stiefel is True, the variables will be updated by SGD-G proposed
        as decorrelated weight matrix.

        If stiefel is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        stiefel (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case stiefel is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case stiefel is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

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
        super(SGDGMuon, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(SGDGMuon, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Debug: remove the optimizer to Check the loss change ==============
        # return loss
        # end debug ====================================================

        for group in self.param_groups:
            stiefel = group["stiefel"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                assert stiefel, "We only target orthogonal matrix in SGDGMuon"
                
                if stiefel:
                    weight_decay = group["weight_decay"]
                    dampening = group["dampening"]
                    nesterov = group["nesterov"]

                    rand_num = random.randint(1, 101)

                    g = p.grad.data.view(p.size()[0], p.size()[1], -1)

                    lr = group["lr"]

                    param_state = self.state[p]
                            
                    p_data = p.data.clone()
                    O = msign(skew(torch.bmm(p_data.transpose(-1, -2), g)))
                    I = torch.eye(p_data.shape[-1]).to(p_data.device).to(p_data.dtype)
                    W1 = torch.bmm(p_data, (I - lr * O))
                    W2 = torch.bmm(W1, (I - torch.bmm(O.transpose(-1, -2), O) * (1 - (1 + lr**2)**(-0.5))))
                    
                    p.data.copy_(W2)


        return loss
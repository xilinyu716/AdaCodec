# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import math

import torch
import transformers

from train_utils.quant_linear import QuantizeLinear
from utils import hadamard_utils
from utils.utils import HadamardTransform
from torch import utils
from utils.hadamard_utils import block_diag_left_matmul, block_diag_matmul


def get_minq_maxq(bits, sym):
    
    assert sym, "We only deal with symmetric quantization"
    if sym:
        maxq = torch.tensor(6.0)
        minq = torch.tensor(-6.0)
    else:
        
        quant_grid = torch.tensor([0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]).to(torch.bfloat16)
        assert bits == 4
        maxq = torch.amax(torch.abs(quant_grid))
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    
    scale = scale.to(x.device)
    quant_grid = torch.tensor([0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]).to(torch.bfloat16).to(x.device)
    
    labels = ((x / scale).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    q = quant_grid[labels].clone() * scale
    # q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

@torch.no_grad
def grids_e4m3(device, dtype):
    lst = []
    b = 7
    for e in range(16):
        for m in range(8):
            for s in [-1, 1]:
                lst.append(s * 2**(e - b) * (1 + m / 8))
    return torch.tensor(lst, device=device, dtype=dtype).contiguous()

@torch.no_grad()
def round_to_e4m3(x):
    grids = grids_e4m3(x.device, x.dtype)
    dist = torch.abs(x.unsqueeze(-1) - grids)
    indices = torch.argmin(dist, dim=-1)
    return grids[indices]
    

@torch.no_grad()
def quant_fp4(x, scaling_factor):
    # x: [batch, seq, hidden] or [-1, hidden]
    # return qx: [batch, seq, hidden] or [-1, hidden]
    
    # print(f"[debug]: quant_fp4")
    init_shape = x.shape
    group_size = 16
    x = x.reshape(-1, min(group_size, x.shape[-1]))

    q_x = (x / scaling_factor).to(torch.half)
    x_bi = q_x.view(torch.short)
    
    x_sign = x_bi & 0x8000
    x_exp_m = x_bi & 0x7FFF
    # if x_exp_m < 0x3400:
    #     # x < 0.25
    #     q_x = 0x0000
    # elif x_exp_m < 0x3A00:
    #     # x < 0.75
    #     q_x = 0x3800
    # elif x_exp_m < 0x3C00:
    #     # x < 1.0
    #     q_x = 0x3C00
    # elif x_exp_m > 0x4600:
    #     # x > 6.0
    #     q_x = 0x4600
    # else:
    #     # x >= 1.0
    #     pass
    x_branch_0_mask = torch.where(x_exp_m >= 0x3400, 0x0001, 0x0000).to(torch.short)
    x_exp_m = x_exp_m * x_branch_0_mask
    x_branch_0 = (0x0001 - x_branch_0_mask) * 0x0000
    
    x_branch_1_mask = torch.where(x_exp_m >= 0x3A00, 0x0001, 0x0000).to(torch.short)
    x_exp_m = x_exp_m * x_branch_1_mask
    x_branch_1 = (x_branch_0_mask - x_branch_1_mask) * 0x3800
    
    x_branch_2_mask = torch.where(x_exp_m >= 0x3C00, 0x0001, 0x0000).to(torch.short)
    x_exp_m = x_exp_m * x_branch_2_mask
    x_branch_2 = (x_branch_1_mask - x_branch_2_mask) * 0x3C00
    
    x_branch_3_mask = torch.where(x_exp_m >= 0x4600, 0x0001, 0x0000).to(torch.short)
    x_exp_m = x_exp_m * (x_branch_2_mask - x_branch_3_mask)
    x_branch_3 = x_branch_3_mask * 0x4600
    
    
    
    x_branch_4 = x_exp_m
    
    x_u = x_branch_4 & 0x7E00
    x_r = x_branch_4 & 0x0100
    
    x_branch_4 = x_u + 0x0002 * x_r
    
    x_exp_m = x_branch_0 + x_branch_1 + x_branch_2 + x_branch_3 + x_branch_4
    
    # print(x_exp_m.dtype)
    # print(x_branch_0.shape)
    # print(x_exp_m.shape, flush=True)
    
    q_x = (x_exp_m + x_sign).view(torch.half)
    q_x = q_x.reshape(init_shape)

    return q_x


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, global_max):
        
        scale = scale.to(x.device)
        x = x / global_max
        q = quant_fp4(x, scale)
        init_shape = q.shape
        group_size = 16
        q = q.reshape(-1, group_size)
        
        # end debug ================================================
        
        
        return (scale * q).reshape(init_shape) * global_max
        # return x

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self) -> None:
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = 16

    def free(self) -> None:
        self.zero = None
        self.scale = None
        self.global_max = None

    def forward(self, x):
        x_dtype = x.dtype
        assert self.sym, "Only use sym quant in NVFP4 W4A4"
        if self.bits == 16:
            return x
        elif self.sym:
            return STEQuantize.apply(x, self.scale, self.global_max).to(x_dtype)
        return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # To be deprecated
    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        
        raise NotImplementedError
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self, bits: int, groupsize: int = -1, sym: bool = False, clip_ratio: float = 1.0
    ) -> None:
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = 16
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"

    @ torch.no_grad()
    def find_params_per_token_groupwise(self, x) -> None:
        init_shape = x.shape
        
        assert self.groupsize == 16, "NVFP4 Quantization requires group_size 16"
        reshaped_x = x.reshape(
            -1, self.groupsize
        )

        xmax = torch.amax(reshaped_x, dim=-1, keepdim=True)
        xmin = torch.amin(reshaped_x, dim=-1, keepdim=True)
        if self.sym:
            self.global_max = 1 / 6
            scales = (torch.amax(torch.abs(reshaped_x), dim=-1, keepdim=True) / self.global_max) / 6.0
            
            # print(f"[debug 1]: scales shape: {scales.shape}")
            scales = round_to_e4m3(scales)
            # print(f"[debug 2]: scales shape: {scales.shape}")
            
            self.scale = scales
            self.zero = torch.zeros_like(self.scale)
            
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        # self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        # self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    @torch.no_grad()
    def find_params(self, x) -> None:
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape
        assert self.groupsize == 16, f"NVFP4 Quantization requires groupsize to be 16, while it is {self.groupsize}"

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            # hadamard_utils.cleanup_memory(verbos=False)
            return

        raise NotImplementedError


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registered to rotate the activation before quantization.
    """

    def __init__(self, module: torch.nn.Linear) -> None:
        super(ActQuantWrapper, self).__init__()
        # assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False


    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def forward(self, x, R1=None, R2=None, RM = None, RM_hat = None, transpose_R1=False, transpose_R2=False):
        
        x.requires_grad_(True)
        assert R2 == None or RM == None, "Can't be both self_attn and MLP"
        x_dtype = x.dtype


        # debug: remove RM to see the change in loss =============
        if not (RM == None):
            x = block_diag_matmul(x, RM)
        # end debug ===============================================

            
        if self.quantizer.bits < 16:  # Quantize, if needed
            # print(f"[debug]: quantize activation")
            with torch.no_grad():
                self.quantizer.find_params(x)
                
            assert self.quantizer.groupsize == 16, "group size must be 16"
            assert self.quantizer.scale.shape[-1] == 1, "Last dimension should be 1"
            assert self.quantizer.scale.numel() == x.numel() // 16, "group size = 16"
            assert self.quantizer.scale.shape[0] == x.numel() // 16, "group size = 16"
            
            # debug quantizer: see how the loss changes =================
           
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()
            
            # end debug =======================================================
            
        # print("Check R1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(R1 is None)
        # print("Check bits")
        # print(self.quantizer.bits)
        

        
        if R1 is not None:
            x = self.module(input = x, R1 = R1, R2 = R2, RM = RM_hat, transpose_R1=transpose_R1, transpose_R2=transpose_R2).to(x_dtype)
        else:
            # print(f"[debug]: No rotation")
            x = self.module(x).to(x_dtype)

        

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            with torch.no_grad():
                self.out_quantizer.find_params(x)
                
            # debug quantizer: see how the loss changes
            
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape: int = 1) -> None:
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
        weight_groupsize: int = -1,
    ) -> None:
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.weight_groupsize = weight_groupsize
        assert sym, "NVFP4 Quantization is symmetric"
        if sym:
            self.maxq = torch.tensor(6.0)
        else:
            self.maxq = torch.tensor(6.0)

    @torch.no_grad()
    def find_params_weight_groupwise(self, x) -> None:
        init_shape = x.shape
        reshaped_x = x.reshape(-1, self.weight_groupsize)
        

        # xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        # xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            global_max = 1 / 6
            
            scales = (torch.amax(torch.abs(reshaped_x), dim=-1, keepdim=True) / global_max) / 6.0
            scales = round_to_e4m3(scales)
            
            # print(f"[debug]: scales = {scales}")
            
            self.scale = scales
            self.global_max = global_max
            self.zero = torch.zeros_like(self.scale)
        else:
            raise NotImplementedError

    @ torch.no_grad()
    def find_params(self, x) -> None:
        if self.bits == 16:
            return
        
        assert self.weight_groupsize == 16, f"NVFP4 Quant require group_size to be 16, but it's now {self.weight_groupsize}"
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape

        if self.weight_groupsize > 0:
            self.find_params_weight_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return
        elif self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            assert self.sym, "Only use symmetric quant in W4A4 NVFP4"
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.global_max).to(x_dtype)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(
                x_dtype
            )
        return x

    # Return MXFP4 value and scale in addtional to fake quantized weight
    def fake_quantize(self, x):

        
        init_shape = x.shape
        assert self.ready() and self.bits < 16
        
        global_scale = self.global_max

        scale = self.scale.to(x.device) * global_scale
        q = quant_fp4(x, scale)
        
        # end debug ================================================
        
        
        return (scale * q.reshape(-1, min(q.shape[-1], 16))).reshape(init_shape), q, scale

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(
    module: ActQuantWrapper,
    name: str = "",
    layers=[
        torch.nn.Linear,
        QuantizeLinear,
        ActQuantWrapper,
        transformers.models.falcon.modeling_falcon.FalconLinear,
    ],
) -> None:
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) is torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) is torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + "." + name1 if name != "" else name1, layers)


def find_qlayers(
    module,
    layers=[torch.nn.Linear, ActQuantWrapper, QuantizeLinear],
    name: str = "",
):
    # fix for llama embedding layer
    if type(module) in [torch.nn.Embedding] and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res

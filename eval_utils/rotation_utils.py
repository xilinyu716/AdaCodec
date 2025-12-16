# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import torch
import tqdm
from torch import nn

from utils import monkeypatch, quant_utils, utils
from utils.quant_utils import ActQuantWrapper
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
)
from utils.utils import HadamardTransform

# TODO: modulize these functions and classes

def block_diag_matmul(weight: torch.Tensor, R_blocks: torch.Tensor) -> torch.Tensor:
    """
    Args:
        weight: Tensor, shape (out_dim, hidden_size)
        R_blocks: Tensor, shape (n_blocks, block_size, block_size)

    Returns:
        Tensor, shape (out_dim, hidden_size)
    """
    
    assert weight.dim() in [2, 3]
    if weight.dim() == 2:
        out_dim, hidden_size = weight.shape
        b = 1
    else:
        b, out_dim, hidden_size = weight.shape
        
    init_shape = weight.shape
    n_blocks, block_size, _ = R_blocks.shape
    assert hidden_size == n_blocks * block_size, \
        f"weight.shape[1]={hidden_size} must equal n_blocks*block_size={n_blocks*block_size}"

    # reshape: (out_dim, hidden_size) -> (out_dim, n_blocks, block_size)
    W_blocks = weight.view(b * out_dim, n_blocks, block_size)
    
    # W @ R_blocks
    R_blocks = R_blocks.to(W_blocks.device).to(W_blocks.dtype)
    W_rot = torch.einsum('onb,nbm->onm', W_blocks, R_blocks)
    
    return W_rot.reshape(init_shape)



def block_diag_left_matmul(R_blocks: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """    
    Args:
        R_blocks: Tensor, shape (n_blocks, block_size, block_size)
        weight: Tensor, shape (hidden_size, out_dim)
        
    Returns:
        Tensor, shape (hidden_size, out_dim)
    """
    
    assert weight.dim() in [2, 3]
    if weight.dim() == 2:
        out_dim, hidden_size = weight.shape
        b = 1
    else:
        b, out_dim, hidden_size = weight.shape
        
    init_shape = weight.shape
    
    n_blocks, block_size, _ = R_blocks.shape
    # hidden_size, out_dim = weight.shape
    assert out_dim == n_blocks * block_size, \
        f"weight.shape[0]={hidden_size} must equal n_blocks*block_size={n_blocks*block_size}"
    
    # reshape: (hidden_size, out_dim) -> (n_blocks, block_size, out_dim)
    W_blocks = weight.view(b, n_blocks, block_size, hidden_size)
    
    # R_i^T @ W_i
    R_blocks = R_blocks.to(W_blocks.device).to(W_blocks.dtype)
    
    RW_blocks = torch.einsum('nom,bnok->bnmk', R_blocks, W_blocks)
    
    return RW_blocks.reshape(init_shape)


class RotateModule(nn.Module):
    def __init__(self, R_init:torch.Tensor):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.clone().to(torch.float32).to(torch.device("cuda")), requires_grad=True)

    def forward(self, x, transpose=False):
        
        raise NotImplementedError
    
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = block_diag_matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
         
        # W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)
        W.weight.data = block_diag_matmul(W_, R1).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, R1) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = block_diag_left_matmul(R1, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        raise NotImplementedError
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = block_diag_matmul(W_, R1).to(device="cpu", dtype=dtype)

def smooth_mod_weight(layer, smooth):
    assert isinstance(layer, nn.Linear)
    dtype = layer.weight.data.dtype
    W_ = layer.weight.data.to(device="cuda", dtype=torch.float64)
    W_ = (W_ * torch.exp(smooth).to(W_.dtype).to(W_.device)).to(device="cpu", dtype=dtype)
    layer.weight.data = W_
    
def rotate_mlp_output(layer, RM):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    
    W_ = block_diag_matmul(W_, RM).to(device="cpu", dtype=dtype)
    
    W.weight.data = W_
    
    # debug: remove R3 to check the PPL without H =========================
    # apply_exact_had_to_linear(
    #     W, had_dim=-1, output=False
    # )  # apply exact (inverse) hadamard on the weights of mlp output
    # end debug ===========================================================
    if W.bias is not None:
        raise NotImplementedError
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = block_diag_matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None, R2_hat = None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2_hat)
    
    
def rotate_R1_layerwise(layer, head_num, head_dim, 
                        R1_q=None, R1_k = None,
                        R1_v=None, R1_o = None,
                        R1_up=None, R1_down = None,
                        R1_gate=None):
    q_proj = layer.self_attn.q_proj
    v_proj = layer.self_attn.v_proj
    k_proj = layer.self_attn.k_proj
    
    o_proj = layer.self_attn.o_proj
    
    up_proj = layer.mlp.up_proj
    gate_proj = layer.mlp.gate_proj
    
    down_proj = layer.mlp.down_proj

    # TODO Deprecate R2 and use common R
    apply_exact_had_to_linear(q_proj, had_dim=head_dim, output=False, R2=R1_q)
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=False, R2=R1_v)
    apply_exact_had_to_linear(k_proj, had_dim=head_dim, output=False, R2=R1_k)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=True, R2=R1_o)
    
    apply_exact_had_to_linear(up_proj, had_dim=head_dim, output=False, R2=R1_up)
    apply_exact_had_to_linear(gate_proj, had_dim=head_dim, output=False, R2=R1_gate)
    apply_exact_had_to_linear(down_proj, had_dim=head_dim, output=True, R2=R1_down)
    
    


@torch.inference_mode()
def rotate_model(model, args):
    R1 = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    if args.optimized_rotation_path is not None:
        R_cpk:str = args.optimized_rotation_path
        R_dict = torch.load(R_cpk)
        print(f"[debug]: R dict = {R_dict}")
        R1 = torch.load(R_cpk)["R1"].cuda().to(torch.float64)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
       # TODO Implement it as exec
        if args.optimized_rotation_path is not None:
            key_R2 = f"model.layers.{idx}.self_attn.R2"
            key_R2_hat = f"model.layers.{idx}.self_attn.R2_hat"
            R2 = torch.load(R_cpk)[key_R2].cuda().to(torch.float64)
            R2_hat = torch.load(R_cpk)[key_R2_hat].cuda().to(torch.float64)
            
            key_R1_q = f"model.layers.{idx}.self_attn.R1_q"
            key_R1_k = f"model.layers.{idx}.self_attn.R1_k"
            key_R1_v = f"model.layers.{idx}.self_attn.R1_v"
            key_R1_o = f"model.layers.{idx}.self_attn.R1_o"
            key_R1_up = f"model.layers.{idx}.mlp.R1_up"
            key_R1_down = f"model.layers.{idx}.mlp.R1_down"
            key_R1_gate = f"model.layers.{idx}.mlp.R1_gate"

            R1_q = torch.load(R_cpk)[key_R1_q].cuda().to(torch.float64)
            R1_k = torch.load(R_cpk)[key_R1_k].cuda().to(torch.float64)
            R1_v = torch.load(R_cpk)[key_R1_v].cuda().to(torch.float64)
            R1_o = torch.load(R_cpk)[key_R1_o].cuda().to(torch.float64)
            R1_up = torch.load(R_cpk)[key_R1_up].cuda().to(torch.float64)
            R1_down = torch.load(R_cpk)[key_R1_down].cuda().to(torch.float64)
            R1_gate = torch.load(R_cpk)[key_R1_gate].cuda().to(torch.float64)
            

        else:
            raise NotImplementedError
            R2 = get_orthogonal_matrix(head_dim, args.rotate_mode)
            
        if args.optimized_rotation_path is not None:
            key_RM = f"model.layers.{idx}.mlp.RM"
            key_RM_hat = f"model.layers.{idx}.mlp.RM_hat"
            RM = torch.load(R_cpk)[key_RM].cuda().to(torch.float64)
            RM_hat = torch.load(R_cpk)[key_RM_hat].cuda().to(torch.float64)
            layers[idx].mlp.RM = RM
        else:
            raise NotImplementedError
            RM = get_orthogonal_matrix(head_dim, args.rotate_mode)
            
        rotate_mlp_output(layers[idx], RM_hat)
        rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2, R2_hat = R2_hat)
        rotate_R1_layerwise(layers[idx], num_heads, head_dim, 
                            R1_q = R1_q, 
                            R1_k = R1_k,
                            R1_v = R1_v,
                            R1_o = R1_o,
                            R1_up = R1_up,
                            R1_down = R1_down,
                            R1_gate = R1_gate)
                    
                
        


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)

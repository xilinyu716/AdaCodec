# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch._tensor import Tensor

# TODO: Move this to rotation_utils
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
        f"weight.shape[0]={out_dim} must equal n_blocks*block_size={n_blocks*block_size}"
    
    # reshape: (hidden_size, out_dim) -> (n_blocks, block_size, out_dim)
    W_blocks = weight.view(b, n_blocks, block_size, hidden_size)
    
    # R_i^T @ W_i
    RW_blocks = torch.einsum('nom,bnok->bnmk', R_blocks, W_blocks)
    
    return RW_blocks.reshape(init_shape)


class QuantizeLinear(nn.Linear):
    def forward(
        self,
        input: Tensor,
        R1=None,
        R2=None,
        RM=None,
        transpose_R1=False,
        transpose_R2=False,
    ) -> Tensor:
        input.requires_grad_(True)
        # quantize weight
        
        # print("In forward!")
        # print(f"R1: {R1}")
        # print(f"R2: {R2}")
        # print(f"weight_dtype: {self.weight.dtype}")
        # print(f"quantizer: {self.quantizer}")
        # print(f"transpose: {transpose}")
        
        # print(f"Input Shape: {input.shape}!!!!!!!!!!!!!!!!!!!!!!!!!!")
        

        assert R2 == None or RM == None, "Can't be both self_attn and mlp"
        
        weight = self.weight.to(torch.float64)
        
        
        # debug: remove the rotation to check the change in loss ===================================
        if R1 is not None:
            dtype = self.weight.dtype
            
            # assert torch.amax(torch.abs(torch.bmm(R1, R1.transpose(-1, -2)) - torch.eye(128).to(R1.device).to(R1.dtype))) < 1e-2, \
            # f"Difference: {torch.amax(torch.abs(torch.bmm(R1, R1.transpose(-1, -2)) - torch.eye(128).to(R1.device).to(R1.device)))}"
            
            # print(f"Is is transpose? {transpose}!!!!!!!!!!!!!!!!!!!!!")
            # exit(0)
            if not transpose_R1:
                
                weight = block_diag_matmul(weight.to(torch.float64), R1.to(torch.float64)).to(
                    dtype
                )
                               
            else:
                
                weight = block_diag_left_matmul(R1.to(torch.float64), weight.to(torch.float64)).to(
                    dtype
                )
                
            if R2 is not None:
                had_dim = R2.shape[-1]
                dtype = weight.dtype
                if transpose_R2:
                    
                    
                    # My Adaptation
                    init_shape = weight.shape
                    weight = block_diag_matmul(weight, R2)
                    weight = weight.reshape(init_shape)
                else:
                    init_shape = weight.shape
                    weight = block_diag_left_matmul(R2, weight)
                    weight = weight.reshape(init_shape)
            weight = weight.to(dtype)
            
            if RM is not None:
                had_dim = RM.shape[-1]
                dtype = weight.dtype
                init_shape = weight.shape
                weight = block_diag_matmul(weight, RM)
                weight = weight.reshape(init_shape)
            weight = weight.to(dtype)
        else:
            weight = self.weight
            
        
        # print(f"[debug]: Is training? {self.training}")
        if hasattr(self, "quantizer") and (not self.training):
            # print(f"[debug]: No rotation, quantize weight")
            dtype = weight.dtype
            
            with torch.no_grad():
                self.quantizer.find_params(weight.data)
                
                
            # debug quantizer: see how loss changes
            weight = self.quantizer.quantize(weight).to(dtype)
            
        # end debug ========================================================================

        

        
        return nn.functional.linear(input, weight, self.bias)

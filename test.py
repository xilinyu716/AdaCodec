import torch
def quant_mxfp4(x, scaling_factor):
    # x: [batch, seq, hidden]
    # return qx: [batch, seq, hidden]
    init_shape = x.shape
    group_size = 32
    x = x.reshape(-1, group_size)
    # print("========================================================")
    # print(f"Input Shape: {x.shape}!!!!!!!!!!!!!!!!!!!")
    # print(f"scaling factor shape: {scaling_factor.shape}!!!!!!!!!!!!!!!!!!!!!!")
    # if (len(init_shape) == 3):
    #     print("Activation Quant")
    # else:
    #     print("Weight Quant")
    # print("========================================================")
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


x = torch.tensor([6] * 32)
print(quant_mxfp4(x, torch.tensor(2.0)))


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


x = torch.tensor([100] * 32, dtype=torch.bfloat16).reshape(2, 16)
print(x)
print(grids_e4m3(x.device, x.dtype))
print(round_to_e4m3(x))

from utils.hadamard_utils import block_dct_tensor

x = block_dct_tensor(32, block_size = 16)
I = torch.bmm(x, x.transpose(-1, -2))
print(I)

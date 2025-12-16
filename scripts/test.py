import torch

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
        f"weight.shape[0]={hidden_size} must equal n_blocks*block_size={n_blocks*block_size}"
    
    # reshape: (hidden_size, out_dim) -> (n_blocks, block_size, out_dim)
    W_blocks = weight.view(b, n_blocks, block_size, hidden_size)
    
    # R_i^T @ W_i
    RW_blocks = torch.einsum('nom,bnok->bnmk', R_blocks, W_blocks)
    
    return RW_blocks.reshape(init_shape)


def random_orthogonal_batch(batch, n):
    A = torch.randn(batch, n, n)
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    Q = Q * d.sign().unsqueeze(-2)
    return Q


Q = random_orthogonal_batch(8, 32)
A = torch.randn(4, 256, 256)
B = torch.randn(4, 256, 256)
AB = torch.bmm(A, B)
AB_hat = torch.bmm(block_diag_matmul(A, Q), block_diag_left_matmul(Q, B))
print(torch.amax(torch.abs(AB - AB_hat)))

AQ = block_diag_matmul(A, Q)
A_hat = block_diag_matmul(AQ, Q.transpose(-2, -1))
print(torch.amax(torch.abs(A - A_hat)))



print(f"Q: {Q}")
A = torch.randn(4, 256, 256)
B = torch.randn(256, 256)
AB_t = torch.nn.functional.linear(A, B)
AB_t_hat = torch.nn.functional.linear(block_diag_matmul(A, Q), block_diag_matmul(B, Q))
print(torch.amax(torch.abs(AB_t - AB_t_hat)))


A = torch.randn(4, 4)
print(torch.max(A))
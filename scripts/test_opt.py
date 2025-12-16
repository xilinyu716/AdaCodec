import torch

def qr_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
    [p, n] = tan_vec.size()
    tan_vec.t_()
    q, r = torch.linalg.qr(tan_vec)
    print(f"q: {q}")
    print(f"r: {r}")
    d = torch.diag(r, 0)
    print(f"d: {d}")
    ph = d.sign()
    print(f"ph: {ph}")
    print(f"factor: {ph.expand_as(q)}")
    q *= ph.expand_as(q)
    q.t_()

    return q

def qr_retraction_3d(tan_vec):
    """
    Batch QR retraction for Stiefel manifold.
    tan_vec: (B, p, n), p <= n
    returns: (B, p, n)
    """
    tan_vec = tan_vec.transpose(-1, -2)  # (B, n, p)
    q, r = torch.linalg.qr(tan_vec)      # q: (B, n, n), r: (B, n, p)
    print(f"q: {q}")
    print(f"r: {r}")
    d = torch.diagonal(r, dim1=-2, dim2=-1)  # (B, p)
    print(f"d: {d}")
    ph = d.sign().unsqueeze(-2)               # (B,1,p)
    print(f"ph: {ph}")
    q = q * ph
    q = q.transpose(-1, -2)                  # (B, p, n)
    return q



tan_vec_2d = torch.randn(2, 2, 4)
q = qr_retraction_3d(tan_vec_2d)
print(q)
print(torch.bmm(q, q.transpose(-1, -2)))

def matrix_norm_one(W):
    out = torch.abs(W)
    print(f"abs: {out}")
    out = torch.sum(out, dim=0)
    print(f"sum: {out}")
    out = torch.max(out)
    print(f"max: {out}")
    return out

W = torch.randn(2, 4)
print(matrix_norm_one(W))
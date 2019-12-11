from tensorly.decomposition import tucker, partial_tucker, parafac
import torch
import tensorly as tl
tl.set_backend('pytorch')

eps = 1e-15


def tucker_decomp(W, rank):
    core, [last, first] = partial_tucker(W + eps, modes=[0, 1], ranks=rank, init='random')
    fk = first.t_()
    lk = last
    new_layers = [fk, core, lk]
    return new_layers


def torch_cp_decomp(W, rank):
    last, first, vertical, horizontal = parafac(W, rank=rank, init='random')
    sr = first.t_()
    rt = last
    rr = torch.stack([vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(0, i, 1) for i in range(rank)]).unsqueeze_(
        1)
    return [sr, rr, rt]


def decomp(layer, rank, type='tucker'):
    if type == 'tucker':
        new_layers = tucker_decomp(layer, rank)
    elif type == 'cp':
        new_layers = torch_cp_decomp(layer, rank)
    else:
        raise NotImplementedError
    return new_layers


def doubly_kronecker_factor(A, cin, patch_size):
    R = A.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).contiguous()
    R = R.view(-1, patch_size**2)
    U, S, V = torch.svd(R)
    first_sv = S[0]
    C = torch.sqrt(first_sv)*U[:, 0].view(cin, cin)
    C = (C + C.t())/2
    K = torch.sqrt(first_sv)*V[:, 0].view(patch_size, patch_size)
    K = (K + K.t())/2
    eps = torch.sign(C.sum())
    C *= eps
    K *= eps
    return [C, K]


def get_svd_decomp(W):
    assert W.shape[2]*W.shape[3] == 1, 'UDF decomposition requires a 4d tensor with shape d1 x d2 x 1 x 1!'
    with torch.no_grad():
        cout, cin, _, _ = W.shape
        W_ = W[:, :, 0, 0]
        U, S, V = torch.svd(W_)
        r = min(cout, cin)
        D = S.new_zeros(r, r)
        d = torch.diagonal(D)
        d += S
        D.unsqueeze_(2).unsqueeze_(3)
    return [U, D, V]


def reverse_decomp(R, C, L):
    if len(C.shape) == 2:
        return L @ C @ R
    elif len(C.shape) == 3:
        C = torch.einsum('ijk, jn -> ink', [C, R])
        C = torch.einsum('ni, ijk -> njk', [L, C])
        return C
    elif len(C.shape) == 4:
        C = torch.einsum('ijkl, jn -> inkl', [C, R])
        C = torch.einsum('ni, ijkl -> njkl', [L, C])
        return C
    else:
        raise NotImplementedError

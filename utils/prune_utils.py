import torch.nn.functional as F
from utils.common_utils import *

eps = 1e-15


def get_threshold(all_importances, prune_ratio):
    all_importances = sorted(all_importances)
    idx = int(prune_ratio*len(all_importances))
    threshold = all_importances[idx]
    return threshold


def filter_indices(channel_saliencies, threshold):
    indices_to_keep = list(np.where(threshold < channel_saliencies)[0])
    if len(indices_to_keep) == 0:
        indices_to_keep = list(np.where(channel_saliencies == channel_saliencies.max())[0])
    return indices_to_keep


def update_module(module):
    if isinstance(module, nn.BatchNorm2d):
        assert hasattr(module, 'in_indices'), '{} has no attribute in_indices!'.format(module)
        module.weight.data = module.weight.data[module.in_indices]
        module.bias.data = module.bias.data[module.in_indices]
        module.running_mean = module.running_mean[module.in_indices]
        module.running_var = module.running_var[module.in_indices]
        module.bias.grad = None
    else:
        assert hasattr(module, 'out_indices'), '{} has no attribute out_indices!'.format(module)
        assert hasattr(module, 'in_indices'), '{} has no attribute in_indices!'.format(module)
        module.weight.data = module.weight.data[module.out_indices][:, module.in_indices]
        if module.bias is not None:
            module.bias.data = module.bias.data[module.out_indices]
            module.bias.grad = None
        if isinstance(module, nn.Conv2d):
            module.in_channels = len(module.in_indices)
            module.out_channels = len(module.out_indices)
        else:
            module.in_features = len(module.in_indices)
            module.out_features = len(module.out_indices)
    module.weight.grad = None


def spread_dependencies_module(module):
    dependencies = module.dependencies.in_
    if len(dependencies) == 0:
        in_indices = list(range(module.weight.data.shape[1]))
    else:
        in_indices = []
        for m_ in dependencies:
            in_indices += m_.out_indices
        in_indices = list(set(in_indices))
    module.in_indices = in_indices


class Dependencies:
    def __init__(self, in_=None, out_=None):
        self.in_ = []
        self.out_ = []
        if in_ is not None:
            self.in_ += in_
        if out_ is not None:
            self.out_ += out_

    def update_in(self, in_):
        self.in_ += in_

    def update_out(self, out_):
        self.out_ += out_


def extract_patches(x, kernel_size, padding, stride):
    assert len(x.shape) == 4, 'extract_patches needs 4d (bs, c, h, w)'
    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    elif isinstance(padding, int):
        pad_h = padding
        pad_w = padding
    else:
        raise NotImplementedError
    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    elif isinstance(stride, int):
        stride_h = stride
        stride_w = stride
    else:
        raise NotImplementedError
    kh, kw = kernel_size
    y = F.pad(x, [pad_w, pad_w, pad_h, pad_h])
    y = y.unfold(2, kh, stride_h).unfold(3, kw, stride_w)
    y = y.transpose(1, 2).transpose(2, 3).contiguous()
    y = y.view(y.size(0), y.size(1)*y.size(2), y.size(3)*y.size(4)*y.size(5))
    return y


def extract_channel_patches(x, kernel_size, padding, stride):
    assert len(x.shape) == 4, 'extract_patches needs 4d (bs, c, h, w)'
    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    elif isinstance(padding, int):
        pad_h = padding
        pad_w = padding
    else:
        raise NotImplementedError
    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    elif isinstance(stride, int):
        stride_h = stride
        stride_w = stride
    else:
        raise NotImplementedError
    kh, kw = kernel_size
    y = F.pad(x, [pad_w, pad_w, pad_h, pad_h])
    y = y.unfold(2, kh, stride_h).unfold(3, kw, stride_w)
    y = y.transpose(1, 2).transpose(2, 3).transpose(3, 4).transpose(4, 5).contiguous()
    y = y.view(y.size(0), y.size(1)*y.size(2), y.size(3)*y.size(4), y.size(5))
    return y


def weight_to_mat(module, use_patch=True, use_bias=True):
    weight = module.weight.data
    if isinstance(module, nn.Conv2d):
        if use_patch:
            weight = weight.transpose(1, 2).transpose(2, 3).contiguous()
            cout = weight.size(0)
            patch_size = weight.size(1)*weight.size(2)
            if use_bias & (module.bias is not None):
                weight = weight.view(-1, weight.size(-1))
                new_col = torch.cat([module.bias.data[k]*module.bias.data.new_ones(patch_size, 1) for k in range(cout)])
                weight = torch.cat([weight, new_col], 1)
            weight = weight.view(cout, patch_size, -1)
        else:
            weight = weight.view(weight.size(0), -1)
            if use_bias & (module.bias is not None):
                new_col = module.bias.unsqueeze(1)
                torch.cat([weight, new_col], 1)
    elif isinstance(module, nn.Linear):
        weight = weight.view(weight.size(0), -1)
        if use_bias & (module.bias is not None):
            new_col = module.bias.unsqueeze(1)
            weight = torch.cat([weight, new_col], 1)
    else:
        raise NotImplementedError
    return weight


def mat_to_weight(module, mat):
    if isinstance(module, nn.Conv2d):
        kh, kw = module.kernel_size
        cin = module.in_channels
        if module.bias is not None:
            weight = mat[:, :-1].view(-1, cin, kh, kw)
            bias = mat[:, -1]
        else:
            weight = mat.view(-1, cin, kh, kw)
            bias = None
    elif isinstance(module, nn.Linear):
        if module.bias is not None:
            weight = mat[:, :-1]
            bias = mat[:, -1]
        else:
            weight = mat
            bias = None
    else:
        raise NotImplementedError
    return weight, bias


def get_inv(M):
    eigval, eigvec = torch.symeig(M, eigenvectors=True)
    eigval[eigval < eps] = eps
    eigval_inv = torch.div(1, eigval)
    M_inv = eigvec @ torch.diag(eigval_inv) @ eigvec.t()
    return M_inv


def get_eigen_decomp(M):
    eigval, eigvec = torch.symeig(M, eigenvectors=True)
    eigval[eigval < eps] = eps
    return eigval, eigvec


class ConvLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=None):
        super(ConvLayerRotation, self).__init__()
        self.bias = bias
        cout, cin = rotation_matrix.size()
        rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3)
        self.conv = nn.Conv2d(out_channels=cout, in_channels=cin, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv.weight.data = rotation_matrix

    def forward(self, x):
        if self.bias is not None:
            x = torch.cat([x, self.bias*x.new_ones(x.size(0), 1, x.size(2), x.size(3))], 1)
        return self.conv(x)


class LinearLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=None):
        super(LinearLayerRotation, self).__init__()
        self.bias = bias
        cout, cin = rotation_matrix.size()
        self.linear = nn.Linear(in_features=cin, out_features=cout, bias=False)
        self.linear.weight.data = rotation_matrix

    def forward(self, x):
        if self.bias is not None:
            x = torch.cat([x, self.bias*x.new_ones(x.size(0), 1)], 1)
        return self.linear(x)


class EigenBasisLayer(nn.Module):
    def __init__(self, Q_G, Q_A, M_new_basis, module, use_bias):
        super(EigenBasisLayer, self).__init__()
        self.sequential = update_layer_basis(module, Q_G, Q_A, M_new_basis, use_bias)

    def forward(self, x):
        return self.sequential(x)


class OriginalBasisLayer(nn.Module):
    def __init__(self, module):
        super(OriginalBasisLayer, self).__init__()
        self.sequential = nn.Sequential(module)

    def forward(self, x):
        return self.sequential(x)


class BasisLayer(nn.Module):
    def __init__(self, module):
        super(BasisLayer, self).__init__()
        self.basis = OriginalBasisLayer(module)

    def forward(self, x):
        return self.basis(x)


def update_layer_basis(module, Q_G, Q_A, M_new_basis, use_bias):
    if isinstance(module, nn.Conv2d):
        patch_size = M_new_basis.size(1)
        if use_bias:
            bias = 1/patch_size
        else:
            bias = None
        rotation_conv_A = ConvLayerRotation(Q_A.t(), bias=bias)
        rotation_conv_G = ConvLayerRotation(Q_G)
        M_new_basis = M_new_basis.view(M_new_basis.size(0), -1, module.kernel_size[0], module.kernel_size[1])
        cout, cin, kh, kw = M_new_basis.size()
        conv_new_basis = nn.Conv2d(out_channels=cout, in_channels=cin, kernel_size=(kh, kw), stride=module.stride,
                                   padding=module.padding, bias=False)
        conv_new_basis.weight.data = M_new_basis
        return nn.Sequential(
            rotation_conv_A,
            conv_new_basis,
            rotation_conv_G
        )
    elif isinstance(module, nn.Linear):
        if use_bias:
            bias = 1
        else:
            bias = None
        rotation_linear_A = LinearLayerRotation(Q_A.t(), bias=bias)
        rotation_linear_G = LinearLayerRotation(Q_G)
        cout, cin = M_new_basis.size()
        linear_new_basis = nn.Linear(out_features=cout, in_features=cin, bias=False)
        linear_new_basis.weight.data = M_new_basis
        return nn.Sequential(
            rotation_linear_A,
            linear_new_basis,
            rotation_linear_G
        )
    else:
        raise NotImplementedError


def regularization_factor_1(A, G):
    trA = torch.trace(A) + eps
    trG = torch.trace(G) + eps
    factor_a = trA/A.size(0)
    factor_g = trG/G.size(0)
    res = factor_a/factor_g
    return torch.sqrt(res)


def regularization_factor_2(C, K, G):
    trC = torch.trace(C) + eps
    trK = torch.trace(K) + eps
    trG = torch.trace(G) + eps
    factor_c = trC/C.size(0)
    factor_k = trK/K.size(0)
    factor_g = trG/G.size(0)
    res_c = ((factor_c**2)/(factor_k * factor_g))**(1/3)
    res_k = ((factor_k**2)/(factor_c * factor_g))**(1/3)
    res_g = ((factor_g**2)/(factor_k * factor_c))**(1/3)
    return [res_c, res_k, res_g]


def get_gradient(module, input, grad_output):
    if isinstance(module, nn.Conv2d):
        a = extract_channel_patches(input, module.kernel_size, module.padding, module.stride)
        batch_size = a.size(0)
        spatial_size = a.size(1)
        patch_size = a.size(2)
        if module.bias is not None:
            a = a.view(-1, a.size(-1))
            a = torch.cat([a, a.new_ones(a.size(0), 1)], 1)
        a = a.view(batch_size, spatial_size, patch_size, -1)
        g = grad_output.transpose(1, 2).transpose(2, 3).contiguous()
        g = g.view(g.size(0), -1, g.size(-1))
        grad = torch.einsum('ijk,ijlm -> iklm', [g, a])
    elif isinstance(module, nn.Linear):
        a = input
        if module.bias is not None:
            a = torch.cat([a, a.new_ones(a.size(0), 1)], 1)
        g = grad_output
        grad = torch.einsum('ij, ik -> ijk', [g, a])
    else:
        raise NotImplementedError
    grad *= input.size(0)
    return grad


def correct_eigen_values(module, gradient, Q_A, Q_G):
    if isinstance(module, nn.Conv2d):
        s_a = gradient @ Q_A
        s = Q_G.t() @ s_a.view(gradient.size(0), gradient.size(1), -1)
        s = s.view(gradient.size())
        s = (s**2).mean(dim=0)
    elif isinstance(module, nn.Linear):
        s = Q_G.t() @ gradient @ Q_A
        s = (s**2).mean(dim=0)
    else:
        raise NotImplementedError
    s[s < eps] = eps
    return s


def get_activation_subfactors(a):
    "a shoud be of size bs x hw x khkw x cin"
    c = a.view(-1, a.size(-1))
    k = a.transpose(2, 3).contiguous()
    k = k.view(-1, k.size(-1))
    return [c, k]


class Pool:
    def __init__(self, pool_momentum, nb_modules, ratio):
        self.pool_momentum = pool_momentum
        self.ratio = ratio
        self.nb_modules = nb_modules
        self.rows = {'score': {}, 'nb': {}}
        self.cols = {'score': {}, 'nb': {}}
        self.rows_score = self.rows['score']
        self.cols_score = self.cols['score']
        self.nb_rows = self.rows['nb']
        self.nb_cols = self.cols['nb']
        self.current_module = 0
        self.it = 0

    def add(self, row_score, col_score, n_rows, n_cols):
        self.nb_rows[self.current_module] = n_rows
        self.nb_cols[self.current_module] = n_cols
        if self.current_module not in self.rows_score.keys():
            self.rows_score[self.current_module] = row_score
            self.cols_score[self.current_module] = col_score
        else:
            self.rows_score[self.current_module] = self.pool_momentum*self.rows_score[self.current_module] +\
                                                   (1 - self.pool_momentum)*row_score
            self.cols_score[self.current_module] = self.pool_momentum*self.cols_score[self.current_module] +\
                                                   (1 - self.pool_momentum)*col_score
        if self.current_module == self.nb_modules-1:
            self.current_module = 0
            self.it += 1
        else:
            self.current_module += 1

    def remove(self, k):
        self.rows_score.pop(k)
        self.cols_score.pop(k)
        self.nb_rows.pop(k)
        self.nb_cols.pop(k)

    def _get_all_scores(self):
        all_scores = list(self.rows_score.values()) + list(self.cols_score.values())
        all_scores = np.array(all_scores)
        all_scores /= all_scores.sum()
        return all_scores

    def _get_all_nb(self):
        res = sum(list(self.nb_rows.values())) + sum(list(self.nb_cols.values()))
        return res

    def random_pick(self):
        assert self.current_module == 0, 'Pool not completly filled!'
        n_samples = int(self.ratio*self._get_all_nb())
        self.nb_rows = {}
        self.nb_cols = {}
        all_scores = self._get_all_scores()
        chosen = np.random.choice(list(range(len(all_scores))), size=n_samples, p=all_scores)
        selected = {k: {'rows': 0, 'cols': 0} for k in self.rows_score.keys()}
        n = len(self.rows_score)
        for k in chosen:
            if k < n:
                p = list(self.rows_score.keys())[k]
                selected[p]['rows'] += 1
            else:
                p = list(self.rows_score.keys())[k-n]
                selected[p]['cols'] += 1
        return selected

    def clear(self):
        self.rows_score = {}
        self.cols_score = {}
        self.nb_rows = {}
        self.nb_cols = {}
        self.current_module = 0


def add_conv_rows(W, n):
    w = W.contiguous().view(W.size(0), -1)
    last = w[-1, :]
    new_rows = W.new_ones(n, w.size(-1))
    new_rows *= last
    w = torch.cat([w, new_rows], 0)
    w = w.view(w.size(0), W.size(1), W.size(2), W.size(3))
    return w


def add_conv_cols(W, n):
    w = W.transpose(0, 1).contiguous()
    w = w.view(w.size(0), -1)
    last = w[-1, :]
    new_rows = W.new_ones(n, w.size(-1))
    new_rows *= last
    w = torch.cat([w, new_rows], 0)
    w = w.view(w.size(0), W.size(0), W.size(2), W.size(3))
    w = w.transpose(0, 1)
    return w


def add_conv(W, n_cols, n_rows):
    if 0 < n_cols:
        W = add_conv_cols(W, n_cols)
    if 0 < n_rows:
        W = add_conv_rows(W, n_rows)
    return W


def add_linear(W, n_cols, n_rows):
    if 0 < n_rows:
        last = W[-1, :]
        new_rows = W.new_ones(n_rows, W.size(-1))
        new_rows *= last
        W = torch.cat([W, new_rows], 0)
    if 0 < n_cols:
        last = W[:, -1].unsqueeze(1)
        new_cols = W.new_ones(size=(W.size(0), n_cols))
        new_cols *= last
        W = torch.cat([W, new_cols], 1)
    return W


def add_filters(W, n_cols, n_rows):
    if len(W.shape) == 4:
        return add_conv(W, n_cols, n_rows)
    elif len(W.shape) == 2:
        return add_linear(W, n_cols, n_rows)
    else:
        raise NotImplementedError


def load_checkpoint_pruning_old(checkpoint_path, net, use_bias):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.add_basis()
    shapes = []
    for module_name in list(checkpoint.keys()):
        if 'sequential' in module_name:
            shapes.append(checkpoint[module_name].shape)
    to_delete = []
    for module in net.modules():
        if isinstance(module, BasisLayer):
            main_module = module.basis.sequential[0]
            use_bias_module = use_bias & (main_module.bias is not None)
            if isinstance(main_module, nn.Conv2d):
                cout, cin, kh, kw = main_module.weight.shape
                Q_G = torch.rand(cout, cout)
                Q_A = torch.rand(cin, cin)
                M_new_basis = torch.rand(cout, cin, kh, kw)
            else:
                cout, cin = main_module.weight.shape
                Q_G = torch.rand(cout, cout)
                Q_A = torch.rand(cin, cin)
                M_new_basis = torch.rand(cout, cin)
            new_basis_layer = EigenBasisLayer(Q_G, Q_A, M_new_basis, main_module, use_bias=use_bias_module)
            to_delete.append(module.basis)
            module.basis = new_basis_layer
    for m in to_delete:
        m.cpu()
        del m

    interesting_modules = [module for module in expand_model(net) if isinstance(module, (nn.Conv2d, nn.Linear))]
    ct = 0
    for module in interesting_modules:
        module.weight.data = torch.rand(shapes[ct])
        if isinstance(module, nn.Conv2d):
            module.out_channels = shapes[ct][0]
            module.in_channels = shapes[ct][1]
        else:
            module.out_features = shapes[ct][0]
            module.in_features = shapes[ct][1]
        ct += 1
    net.load_state_dict(checkpoint)
    return net


def load_checkpoint_pruning(checkpoint_path, net, use_bias):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.add_basis()
    shapes = []
    l = []
    for module_name in list(checkpoint.keys()):
        if ('sequential' in module_name) & ('bias' not in module_name):
            if '.conv.weight' not in module_name:
                n = int(module_name[-len('.weight')-1])
                l.append(n)
            shapes.append(checkpoint[module_name].shape)
    to_delete = []
    ct = 0
    for module in net.modules():
        if isinstance(module, BasisLayer):
            if l[ct] != 0:
                main_module = module.basis.sequential[0]
                use_bias_module = use_bias & (main_module.bias is not None)
                if isinstance(main_module, nn.Conv2d):
                    cout, cin, kh, kw = main_module.weight.shape
                    Q_G = torch.rand(cout, cout)
                    Q_A = torch.rand(cin, cin)
                    M_new_basis = torch.rand(cout, cin, kh, kw)
                else:
                    cout, cin = main_module.weight.shape
                    Q_G = torch.rand(cout, cout)
                    Q_A = torch.rand(cin, cin)
                    M_new_basis = torch.rand(cout, cin)
                new_basis_layer = EigenBasisLayer(Q_G, Q_A, M_new_basis, main_module, use_bias=use_bias_module)
                to_delete.append(module.basis)
                module.basis = new_basis_layer
            ct += 1
    for m in to_delete:
        m.cpu()
        del m

    interesting_modules = [module for module in expand_model(net) if isinstance(module, (nn.Conv2d, nn.Linear))]
    ct = 0
    for module in interesting_modules:
        module.weight.data = torch.rand(shapes[ct])
        if isinstance(module, nn.Conv2d):
            module.out_channels = shapes[ct][0]
            module.in_channels = shapes[ct][1]
        else:
            module.out_features = shapes[ct][0]
            module.in_features = shapes[ct][1]
        ct += 1
    net.load_state_dict(checkpoint)
    return net
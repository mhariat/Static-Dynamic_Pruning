from prune.low_rank_pruning.low_rank_pruning import *
from utils.decomp_utils import *


class ImprovedEigenPrunerMulti(LowRankPruner):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval, use_bias=True,
                 fisher_type='true', allow_increase=False, regularization=1e-3, ma=0.9, correct_eigenvalues=False,
                 sua=False, use_full_cov_a=True, decomp_method='tucker', pool_momentum=0.5,
                 back_ratio=0.25):
        super(ImprovedEigenPrunerMulti, self).__init__(model=model, prune_ratio_limit=prune_ratio_limit,
                                                       normalize=normalize, log_interval=log_interval, use_hook=True)
        self.use_bias = use_bias
        self.G = OrderedDict()
        self.A = OrderedDict()
        self.C = OrderedDict()
        self.K = OrderedDict()
        self.a = OrderedDict()
        self.g = OrderedDict()
        self.Q_G = OrderedDict()
        self.Q_A = OrderedDict()
        self.Q_K = OrderedDict()
        self.M_new_basis = OrderedDict()
        self.eigen_values = OrderedDict()
        self.regularization = regularization
        self.ma = ma
        self.allow_increase = allow_increase
        self.fisher_type = fisher_type
        self.decomp_method = decomp_method
        self.sua = sua
        self.correct_eigenvalues = correct_eigenvalues
        self.use_full_cov_a = use_full_cov_a
        self.pool_momentum = pool_momentum
        self.back_ratio = back_ratio
        self.brings_back = False
        if 0 < self.back_ratio:
            self.brings_back = True
        self.model.add_basis()
        self.nb_basis_modules = self._get_nb_basis_layer()
        assert self.nb_basis_modules == len(self.interesting_modules),\
            "Number of basis layer and number of interesting modules are not the same!"
        self.pool = Pool(self.pool_momentum, self.nb_basis_modules, self.back_ratio)
        self.multi_gpu = True

    def _get_nb_basis_layer(self):
        ct = 0
        for module in self.model.modules():
            if isinstance(module, BasisLayer):
                ct += 1
        return ct

    def _forward_conv(self, module, input):
        key = (module.idx, input[0].device.index)
        x = extract_channel_patches(input[0].data, module.kernel_size, module.padding, module.stride)
        patch_size = x.size(2)
        batch_size = x.size(0)
        if self.use_bias & (module.bias is not None):
            x = x.view(-1, x.size(-1))
            new_col = x.new_ones(x.size(0), 1) / patch_size
            x = torch.cat([x, new_col], 1)
            x = x.view(batch_size, -1, patch_size, x.size(-1))
        if self.sua | (patch_size == 1):
            a = x.view(-1, x.size(-1))
            value = (a.t() @ a)/batch_size
            if self.steps == 0:
                self.A[key] = value
            else:
                moving_average(self.A[key], value, self.ma)
        elif self.use_full_cov_a:
            a = x.transpose(2, 3).contiguous()
            a = a.view(-1, a.size(2) * a.size(3))
            value = (a.t() @ a)/batch_size
            if self.steps == 0:
                self.A[module] = value
            else:
                moving_average(self.A[module], value, self.ma)
        else:
            c, k = get_activation_subfactors(x)
            value_c = (c.t() @ c) / batch_size
            value_k = (k.t() @ k) / batch_size / patch_size
            d = torch.diagonal(value_k)
            d.fill_(d.mean())
            value_k /= d.mean()
            if self.steps == 0:
                self.C[module] = value_c
                self.K[module] = value_k
            else:
                moving_average(self.C[module], value_c, self.ma)
                moving_average(self.K[module], value_k, self.ma)

    def _forward_linear(self, module, input):
        key = (module.idx, input[0].device.index)
        a = input[0].data
        if self.use_bias & (module.bias is not None):
            new_col = a.new_ones(a.size(0), 1)
            a = torch.cat([a, new_col], 1)
        batch_size = a.size(0)
        value = (a.t() @ a) / batch_size
        if self.steps == 0:
            self.A[key] = value
        else:
            moving_average(self.A[key], value, self.ma)

    def _forward_func(self, module, input, output):
        key = (module.idx, input[0].device.index)
        super(LowRankPruner, self)._forward_func(module, input, output)
        if isinstance(module, nn.Conv2d):
            self._forward_conv(module, input)
        elif isinstance(module, nn.Linear):
            self._forward_linear(module, input)
        else:
            raise NotImplementedError
        if self.steps == 0:
            self.a[key] = input[0].data

    def _backward_func(self, module, grad_wr_input, grad_wr_output):
        key = (module.idx, grad_wr_output[0].device.index)
        super(ImprovedEigenPrunerMulti, self)._backward_func(module, grad_wr_input, grad_wr_output)
        if isinstance(module, nn.Conv2d):
            g = grad_wr_output[0].transpose(1, 2).transpose(2, 3).contiguous()
            spatial_size = g.size(1) * g.size(2)
        elif isinstance(module, nn.Linear):
            g = grad_wr_output[0]
            spatial_size = 1
        else:
            raise NotImplementedError
        batch_size = g.size(0)
        g = g.view(-1, g.size(-1))
        value = (g.t() @ g) * batch_size / spatial_size
        if self.steps == 0:
            self.G[key] = value
            self.g[key] = grad_wr_output[0].data
        else:
            moving_average(self.G[key], value, self.ma)

    def _update_saliencies(self):
        with torch.no_grad():
            for module in self.interesting_modules:
                if isinstance(module, nn.Conv2d):
                    if (module.kernel_size[0]*module.kernel_size[1] == 1) | self.sua:
                        self._update_saliencies_1(module)
                    else:
                        self._update_saliencies_2(module)
                elif isinstance(module, nn.Linear):
                    self._update_saliencies_1(module)
                else:
                    raise NotImplementedError

    def _update_saliencies_1(self, module):
        M = weight_to_mat(module, use_patch=True)
        G = self.G[module]
        A = self.A[module]
        pi = regularization_factor_1(A, G)
        A += torch.sqrt(self.regularization*pi)*torch.diag(pi.new_ones(A.size(0)))
        G += torch.sqrt(self.regularization/pi)*torch.diag(pi.new_ones(G.size(0)))
        l_G, Q_G = get_eigen_decomp(G)
        l_A, Q_A = get_eigen_decomp(A)
        M_new_basis_A = M.view(-1, M.size(-1)) @ Q_A
        M_new_basis_G = Q_G.t() @ M_new_basis_A.view(M.size(0), -1)
        M_new_basis = M_new_basis_G.view(M.size())
        if self.correct_eigenvalues:
            grad_bs = get_gradient(module, self.a[module], self.g[module])
            s = correct_eigen_values(module, grad_bs, Q_A, Q_G)
        else:
            if isinstance(module, nn.Conv2d):
                s = (l_G.unsqueeze(1) @ l_A.unsqueeze(0)).unsqueeze(1)
            elif isinstance(module, nn.Linear):
                s = (l_G.unsqueeze(1) @ l_A.unsqueeze(0))
            else:
                raise NotImplementedError
        saliencies_matrix = s * (M_new_basis ** 2)
        if isinstance(module, nn.Conv2d):
            self.saliencies[module] = saliencies_matrix.sum(dim=1)
            M_new_basis = M_new_basis.transpose(1, 2).contiguous()
            s = s.transpose(1, 2).contiguous()
        elif isinstance(module, nn.Linear):
            self.saliencies[module] = saliencies_matrix
        else:
            raise NotImplementedError
        self.eigen_values[module] = s
        self.Q_A[module] = [Q_A]
        self.Q_G[module] = [Q_G]
        self.M_new_basis[module] = M_new_basis

    def _update_saliencies_2(self, module):
        M = weight_to_mat(module, use_patch=True)
        cout, patch_size, cin = M.size()
        G = self.G[module]
        if self.use_full_cov_a:
            A = self.A[module]
            C, K = doubly_kronecker_factor(A, cin, patch_size)
        else:
            C = self.C[module]
            K = self.K[module]
        pi_c, pi_k, pi_g = regularization_factor_2(C, K, G)
        C += (self.regularization**(1/3)) * pi_c * torch.diag(pi_c.new_ones(C.size(0)))
        K += (self.regularization**(1/3)) * pi_k * torch.diag(pi_k.new_ones(K.size(0)))
        G += (self.regularization**(1/3)) * pi_g * torch.diag(pi_g.new_ones(G.size(0)))
        l_C, Q_C = get_eigen_decomp(C)
        l_K, Q_K = get_eigen_decomp(K)
        l_G, Q_G = get_eigen_decomp(G)
        M_1 = torch.einsum('ikj, kl -> ilj', [M, Q_K])
        M_2 = M_1 @ Q_C
        M_3 = Q_G.t() @ M_2.view(M.size(0), -1)
        M_new_basis = M_3.view(M.size())
        if self.correct_eigenvalues:
            grad_bs = get_gradient(module, self.a[module], self.g[module])
            grad_bs = torch.einsum('bikj, kl -> bilj', [grad_bs, Q_K])
            s = correct_eigen_values(module, grad_bs, Q_C, Q_G)
        else:
            s = torch.einsum('i,j,k -> ijk', [l_G, l_K, l_C])
        saliencies_matrix = s * (M_new_basis ** 2)
        self.saliencies[module] = saliencies_matrix.sum(dim=1)
        M_new_basis = M_new_basis.transpose(1, 2).contiguous()
        s = s.transpose(1, 2).contiguous()
        self.eigen_values[module] = s
        self.Q_A[module] = [Q_C]
        self.Q_G[module] = [Q_G]
        self.Q_K[module] = Q_K
        self.M_new_basis[module] = M_new_basis

    def _compute_saliencies(self, dataloader=None):
        super(ImprovedEigenPrunerMulti, self)._compute_saliencies(dataloader=dataloader)
        self.init_step()
        self.model = nn.DataParallel(self.model.cuda())
        self.model.train()

        for batch_idx, (data, target) in enumerate(dataloader):

            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            if self.skip:
                output, _, _ = self.model(data)
            else:
                output = self.model(data)
            criterion = nn.CrossEntropyLoss(reduction='mean')
            if self.fisher_type == 'true':
                samples = torch.multinomial(output.data.softmax(dim=1), 1).squeeze()
                loss = criterion(output, samples)
            elif self.fisher_type == 'exp':
                loss = criterion(output, target)
            else:
                raise NotImplementedError
            loss.backward()
            self.update_step()
        self.model = self.model.module
        self._treat_order_dict()
        if not self.ma:
            self.step_normalization()
        self._update_saliencies()

    def _treat_order_dict(self):
        new_A = OrderedDict()
        new_G = OrderedDict()
        new_a = OrderedDict()
        new_g = OrderedDict()
        for module in self.interesting_modules:
            new_A[module] = (self.A[(module.idx, 0)] + self.A[(module.idx, 1)].cuda(0))/2
            new_G[module] = (self.G[(module.idx, 0)] + self.G[(module.idx, 1)].cuda(0))/2
            new_a[module] = torch.cat([self.a[(module.idx, 0)], self.a[(module.idx, 1)].cuda(0)])
            new_g[module] = torch.cat([self.g[(module.idx, 0)], self.g[(module.idx, 1)].cuda(0)])
        self.A = new_A
        self.G = new_G
        self.g = new_g
        self.a = new_a

    def _make_changes(self, prune_ratio):
        threshold = self._get_threshold(prune_ratio)
        for module in self.saliencies.keys():
            cout, cin = self.saliencies[module].size()
            row_saliencies = self.saliencies[module].sum(dim=1).cpu().numpy()
            col_saliencies = self.saliencies[module].sum(dim=0).cpu().numpy()
            row_indices = filter_indices(row_saliencies, threshold)
            col_indices = filter_indices(col_saliencies, threshold)
            row_ratio = 1 - len(row_indices)/cout
            col_ratio = 1 - len(col_indices)/cin
            if self.prune_ratio_limit < row_ratio:
                row_threshold = get_threshold(row_saliencies, self.prune_ratio_limit)
                row_indices = filter_indices(row_saliencies, row_threshold)
            if self.prune_ratio_limit < col_ratio:
                col_threshold = get_threshold(col_saliencies, self.prune_ratio_limit)
                col_indices = filter_indices(col_saliencies, col_threshold)
            if self.decomp_method == 'usual':
                self.M_new_basis[module] = self.M_new_basis[module][row_indices, :][:, col_indices]
                self.Q_G[module][0] = self.Q_G[module][0][:, row_indices]
                self.Q_A[module][0] = self.Q_A[module][0][:, col_indices]
            elif self.decomp_method == 'tucker':
                r1 = len(row_indices)
                r2 = len(col_indices)
                W = self.M_new_basis[module]
                right, core, left = tucker_decomp(W, [r1, r2])
                self.Q_G[module][0] @= left
                self.Q_A[module][0] @= right.t()
                self.M_new_basis[module] = core

            else:
                raise NotImplementedError
            deleted_row_indices = [k for k in list(range(len(row_saliencies))) if k not in row_indices]
            deleted_col_indices = [k for k in list(range(len(col_saliencies))) if k not in col_indices]
            deleted_row_saliencies = row_saliencies[deleted_row_indices]
            deleted_col_saliencies = col_saliencies[deleted_col_indices]

            if self.brings_back:
                self._update_pool(deleted_row_saliencies, deleted_col_saliencies)

    def _update_pool(self, deleted_row_saliencies, deleted_col_saliencies):
        n_rows = len(deleted_row_saliencies)
        n_cols = len(deleted_col_saliencies)
        all_saliencies_sum = sum(self.all_saliencies)
        deleted_row_saliencies_score = deleted_row_saliencies.sum()/all_saliencies_sum
        deleted_col_saliencies_score = deleted_col_saliencies.sum()/all_saliencies_sum
        self.pool.add(deleted_row_saliencies_score, deleted_col_saliencies_score, n_rows, n_cols)

    def _get_rotation_matrices(self):
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, BasisLayer):
                    if isinstance(module.basis, EigenBasisLayer):
                        sequential = module.basis.sequential
                        if isinstance(sequential[1], nn.Conv2d):
                            prev_Q_A = sequential[0].conv.weight.data
                            prev_Q_G = sequential[2].conv.weight.data
                        elif isinstance(sequential[1], nn.Linear):
                            prev_Q_A = sequential[0].linear.weight.data
                            prev_Q_G = sequential[2].linear.weight.data
                        else:
                            raise NotImplementedError
                        prev_Q_A = prev_Q_A.view(prev_Q_A.size(0), prev_Q_A.size(1)).transpose(1, 0)
                        prev_Q_G = prev_Q_G.view(prev_Q_G.size(0), prev_Q_G.size(1))
                        self.Q_A[sequential[1]].append(prev_Q_A)
                        self.Q_G[sequential[1]].append(prev_Q_G)

    def _update_rotation_matrices(self):
        with torch.no_grad():
            for module in self.interesting_modules:
                if len(self.Q_A[module]) == 2:
                    self.Q_A[module] = self.Q_A[module][1] @ self.Q_A[module][0]
                    self.Q_G[module] = self.Q_G[module][1] @ self.Q_G[module][0]
                else:
                    self.Q_A[module] = self.Q_A[module][0]
                    self.Q_G[module] = self.Q_G[module][0]

    def _update_network(self):
        self._get_rotation_matrices()
        self._update_rotation_matrices()
        self.interesting_modules = []
        ct = 0
        to_delete = []
        for module in self.model.modules():
            if isinstance(module, BasisLayer):
                if isinstance(module.basis, EigenBasisLayer):
                    main_module = module.basis.sequential[1]
                    use_bias = self.use_bias & (module.basis.sequential[0].bias is not None)
                elif isinstance(module.basis, OriginalBasisLayer):
                    main_module = module.basis.sequential[0]
                    use_bias = self.use_bias & (main_module.bias is not None)
                else:
                    raise NotImplementedError
                Q_G = self.Q_G[main_module]
                Q_A = self.Q_A[main_module]
                M_new_basis = self.M_new_basis[main_module]
                if main_module in self.Q_K.keys():
                    M_new_basis @= self.Q_K[main_module].t()
                new_basis_layer = EigenBasisLayer(Q_G, Q_A, M_new_basis, main_module, use_bias=use_bias)
                new_main_module = new_basis_layer.sequential[1]
                if not self.allow_increase:
                    nb_parameters_new_basis_layer = sum([count_parameters(m) for m in expand_model(new_basis_layer)])
                    nb_parameters_prev_basis_layer = sum([count_parameters(m) for m in expand_model(module.basis)])
                    ratio = nb_parameters_new_basis_layer/nb_parameters_prev_basis_layer
                    if ratio <= 1:
                        to_delete.append(module.basis)
                        module.basis = new_basis_layer
                        new_main_module.idx = ct
                        self.interesting_modules.append(new_main_module)
                    else:
                        main_module.idx = ct
                        self.interesting_modules.append(main_module)
                        if self.brings_back:
                            self.pool.remove(ct)
                else:
                    to_delete.append(module.basis)
                    module.basis = new_basis_layer
                    new_main_module.idx = ct
                    self.interesting_modules.append(new_main_module)
                ct += 1
        for m in to_delete:
            m.cpu()
            del m

    def _bring_back(self):
        selected = self.pool.random_pick()
        ct = 0
        for module in self.model.modules():
            if isinstance(module, BasisLayer):
                if isinstance(module.basis, EigenBasisLayer):
                    n_rows = selected[ct]['rows']
                    n_cols = selected[ct]['cols']
                    sequential = module.basis.sequential
                    main_module = sequential[1]
                    W = main_module.weight.data
                    if isinstance(main_module, nn.Conv2d):
                        Q_A = sequential[0].conv.weight.data
                        Q_G = sequential[2].conv.weight.data
                    else:
                        Q_A = sequential[0].linear.weight.data
                        Q_G = sequential[2].linear.weight.data
                    Q_A = Q_A.view(Q_A.size(0), -1).t()
                    Q_G = Q_G.view(Q_G.size(0), -1)
                    W = add_filters(W, n_cols, n_rows)
                    Q_A = add_filters(Q_A, n_cols, 0)
                    Q_G = add_filters(Q_G, n_rows, 0)
                    self.Q_A[main_module] = Q_A
                    self.Q_G[main_module] = Q_G
                    self.M_new_basis[main_module] = W
                ct += 1

    def _update_network_back(self):
        self.interesting_modules = []
        ct = 0
        to_delete = []
        for module in self.model.modules():
            if isinstance(module, BasisLayer):
                if isinstance(module.basis, EigenBasisLayer):
                    main_module = module.basis.sequential[1]
                    Q_G = self.Q_G[main_module]
                    Q_A = self.Q_A[main_module]
                    M_new_basis = self.M_new_basis[main_module]
                    use_bias = self.use_bias & (module.basis.sequential[0].bias is not None)
                    new_basis_layer = EigenBasisLayer(Q_G, Q_A, M_new_basis, main_module, use_bias=use_bias)
                    to_delete.append(module.basis)
                    module.basis = new_basis_layer
                    new_main_module = new_basis_layer.sequential[1]
                else:
                    new_main_module = module.basis.sequential[0]
                new_main_module.idx = ct
                self.interesting_modules.append(new_main_module)
                ct += 1
        for m in to_delete:
            m.cpu()
            del m

    def _get_all_saliencies(self):
        all_saliencies = []
        for module in self.saliencies.keys():
            row_saliency = self.saliencies[module].sum(dim=1)
            col_saliency = self.saliencies[module].sum(dim=0)
            if self.normalize:
                row_saliency /= row_saliency.sum()
                col_saliency /= row_saliency.sum()
            all_saliencies += list(row_saliency.cpu().numpy()) + list(col_saliency.cpu().numpy())
        return all_saliencies

    def step_normalization(self):
        for module in self.interesting_modules:
            self.G[module] /= self.steps
            self.A[module] /= self.steps

    def prune(self, prune_ratio, train_dataloader):
        if self.pool.it % 2 == 0:
            self._prepare()
            self._compute_saliencies(dataloader=train_dataloader)
            self._make_changes(prune_ratio=prune_ratio)
            self._update_network()
        else:
            self._bring_back()
            self._update_network_back()
            self.pool.it += 1
        self._clean_up()

    def get_type(self):
        res = {}
        ct = 0
        for module in self.model.modules():
            if isinstance(module, BasisLayer):
                if isinstance(module.basis, EigenBasisLayer):
                    res[ct] = "EigenBasis"
                else:
                    res[ct] = "OriginalBasis"
                ct += 1
        print(res)

    def _clear_buffers(self):
        super(ImprovedEigenPrunerMulti, self)._clear_buffers()
        self.G = OrderedDict()
        self.A = OrderedDict()
        self.Q_G = OrderedDict()
        self.Q_A = OrderedDict()
        self.Q_K = OrderedDict()
        self.eigen_values = OrderedDict()
        self.M_new_basis = OrderedDict()
        self.a = OrderedDict()
        self.g = OrderedDict()
        self.C = OrderedDict()
        self.K = OrderedDict()

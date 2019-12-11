from prune.pruning import *


class LowRankPruner(Pruner, metaclass=ABCMeta):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval, use_hook):
        super(LowRankPruner, self).__init__(model=model, prune_ratio_limit=prune_ratio_limit, normalize=normalize,
                                            log_interval=log_interval, use_hook=use_hook)

    def get_nb_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_nb_parameters_per_module(self):
        res = {}
        k = 0
        for module in self.model.modules():
            if isinstance(module, BasisLayer):
                if isinstance(module.basis, OriginalBasisLayer):
                    main_module = module.basis.sequential[0]
                elif isinstance(module.basis, EigenBasisLayer):
                    main_module = module.basis.sequential[1]
                else:
                    raise NotImplementedError
                if isinstance(main_module, nn.Conv2d):
                    key = 'Conv_{}'.format(k)
                elif isinstance(main_module, nn.Linear):
                    key = 'Linear_{}'.format(k)
                else:
                    raise NotImplementedError
                res[key] = sum([count_parameters(m) for m in expand_model(module)])
                k += 1
        return res

    @abstractmethod
    def _compute_saliencies(self, dataloader=None):
        pass

    @abstractmethod
    def _get_all_saliencies(self):
        pass

    @abstractmethod
    def _make_changes(self, prune_ratio):
        pass

    @abstractmethod
    def _update_network(self):
        pass

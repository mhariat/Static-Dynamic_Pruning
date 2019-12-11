from utils.prune_utils import *
from utils.common_utils import *
from models.gates import *


def _weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # nn.init.normal_(m.weight, 0, 0.01)
        # nn.init.normal_(m.weight, 0, 0.001)
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class Models(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(Models, self).__init__()

    def get_modules(self):
        res = []
        for module in expand_model(self):
            if module not in res:
                res.append(module)
        return res

    def get_module_with_dependencies(self):
        return [module for module in self.get_modules() if hasattr(module, 'dependencies')]

    def _spread_dependencies(self):
        for module in self.get_module_with_dependencies():
            spread_dependencies_module(module)

    def _initialize_weights(self):
        self.apply(_weights_init)

    def prune_channels(self):
        self._spread_dependencies()
        for module in self.get_module_with_dependencies():
            update_module(module)

    def get_additional_flops(self):
        return 0

    def get_additional_parameters(self):
        return 0

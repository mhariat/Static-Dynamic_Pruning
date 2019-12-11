from utils.prune_utils import *
from collections import OrderedDict
import copy


class Pruner(metaclass=ABCMeta):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval, use_hook):
        self.model = model
        self.original_model = copy.deepcopy(model).cpu()
        self.prune_ratio_limit = prune_ratio_limit
        self.normalize = normalize
        self.log_interval = log_interval
        self.use_hook = use_hook
        self.all_modules = self.model.get_modules()
        self.steps = 0
        self.use_cuda = torch.cuda.is_available()
        self.saliencies = OrderedDict()
        self.back_hooks = OrderedDict()
        self.for_hooks = OrderedDict()
        self.know_modules = (nn.Conv2d, nn.Linear)
        self.interesting_modules = self._get_interesting_modules()
        self.extra_stat = {}
        self.saved_model = []
        self.it = 0

    @abstractmethod
    def get_nb_parameters(self):
        pass

    @abstractmethod
    def get_nb_parameters_per_module(self):
        pass

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

    def _get_interesting_modules(self):
        ct = 0
        res = []
        for module in self.all_modules:
            if isinstance(module, self.know_modules):
                if not hasattr(module, 'not_interesting'):
                    module.idx = ct
                    res.append(module)
                    ct += 1
        return res

    def init_step(self):
        self.steps = 0

    def update_step(self):
        self.steps += 1

    def update_it(self):
        self.it += 1

    def _forward_func(self, module, input, output):
        pass

    def _backward_func(self, module, grad_wr_input, grad_wr_output):
        pass

    def _update_extra_stat(self, d):
        self.extra_stat.update(d)

    def _launch_hook(self):
        for module in self.interesting_modules:
            self.back_hooks[module] = module.register_forward_hook(self._forward_func)
            self.for_hooks[module] = module.register_backward_hook(self._backward_func)

    def _stop_hook(self):
        for module in self.back_hooks.keys():
            self.back_hooks[module].remove()
            self.for_hooks[module].remove()

    def _clear_buffers(self):
        self.saliencies = OrderedDict()
        self.back_hooks = OrderedDict()
        self.for_hooks = OrderedDict()

    def _prepare(self):
        self.init_step()
        self.extra_stat = {}
        if self.use_hook:
            self._launch_hook()

    def _get_threshold(self, prune_ratio):
        self.all_saliencies = self._get_all_saliencies()
        threshold = get_threshold(self.all_saliencies, prune_ratio)
        return threshold

    def _clean_up(self):
        if self.use_hook:
            self._stop_hook()
        self._clear_buffers()
        torch.cuda.empty_cache()
        gc.collect()

    def prune(self, prune_ratio, train_dataloader):
        self._prepare()
        if self.use_hook:
            self._compute_saliencies(dataloader=train_dataloader)
        else:
            self._compute_saliencies()
        self._make_changes(prune_ratio=prune_ratio)
        self._update_network()
        self._clean_up()

    def fine_tune(self, epochs, optimizer, scheduler, train_dataloader, val_dataloader, alpha, logger=None):
        starting_lr = optimizer.param_groups[0]['lr']
        lr = starting_lr
        multi_gpu = False
        if hasattr(self, 'multi_gpu'):
            multi_gpu = True
        for epoch in range(epochs):
            train_accuracy, train_loss = train_sp(model=self.model, optimizer=optimizer, use_cuda=self.use_cuda,
                                                  train_dataloader=train_dataloader, epoch=epoch,
                                                  log_interval=self.log_interval, multi_gpu=multi_gpu, alpha=alpha)
            val_accuracy, val_loss, skip_ratios = validation_sp(model=self.model, use_cuda=self.use_cuda,
                                                                val_dataloader=val_dataloader)
            skip_summaries = []
            for idx in range(skip_ratios.len):
                skip_summaries.append(1 - skip_ratios.avg[idx])
            cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
            message = 'Fine-tuning. Epoch: [{}/{}]. Train Loss: {:.6f}. Train Accuracy: {:.2f}%.' \
                      ' Validation loss: {:.6f}. Validation Accuracy: {}/{} ({:.2f}%).' \
                      ' Current learning rate: {:.4f}. (started at: {:.4f}). Computation Percentage: {:.3f} %'. \
                format(epoch + 1, epochs, train_loss, 100 * train_accuracy, val_loss,
                       int(val_accuracy * len(val_dataloader.dataset)), len(val_dataloader.dataset),
                       100 * val_accuracy, lr, starting_lr, cp)
            message = '[Iteration: {}]. {}'.format(self.it, message)
            print(colored('\n{}\n'.format(message), 'yellow'))
            if logger is not None:
                logger.info(message)
            lr = scheduler.update_lr(epoch)

        self._update_extra_stat({'Indicator/Epochs': epochs})
        self.update_it()
        torch.cuda.empty_cache()
        gc.collect()

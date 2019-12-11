from utils.compute_flops import *
from utils.compute_time import *
from torch.utils.tensorboard import SummaryWriter
from utils.log import *
import json


class PruneResults:
    def __init__(self, result_dir, train_dataloader, val_dataloader, pruner, exp_name=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.pruner = pruner
        if exp_name is None:
            exp_name = 'unknown_exp'
        model_name = self.pruner.model.name
        depth_model = self.pruner.model.depth
        dataset = self.pruner.dataset
        path_to_add = '{}/{}/{}/{}/{}'.format(dataset, model_name, depth_model, self.pruner.name, exp_name)
        writer_dir = '{}/writer/pruned/{}'.format(result_dir, path_to_add)
        run = get_run(writer_dir, pattern='run')
        file_name = 'run_{}'.format(run)
        self.file_name = file_name
        writer_dir = '{}/{}'.format(writer_dir, self.file_name)
        create_dir(writer_dir)
        self.writer = SummaryWriter(writer_dir)
        self.writer = SummaryWriter(writer_dir)
        checkpoint_dir = '{}/checkpoint/pruned/{}'.format(result_dir, path_to_add)
        create_dir(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        stats_dir = '{}/stats/pruned/{}'.format(result_dir, path_to_add)
        create_dir(stats_dir)
        log_dir = '{}/logs/pruned/{}'.format(result_dir, path_to_add)
        logger = get_logger('log_{}'.format(self.file_name), log_dir)
        pruner.logger = logger
        self.nb_classes = len(train_dataloader.dataset.classes)
        self.logger = logger
        self.stats_dir = stats_dir
        self.stats = {}
        self.steps = 0
        self.prune_steps = 0
        self.add_results()

    def _compute_basic_stats(self):
        block_print()
        cp = 0
        train_acc, train_loss, _ = validation_sp(model=self.pruner.model, use_cuda=torch.cuda.is_available(),
                                                 val_dataloader=self.train_dataloader)
        val_acc, val_loss, skip_ratios = validation_sp(model=self.pruner.model, use_cuda=torch.cuda.is_available(),
                                                       val_dataloader=self.val_dataloader)
        skip_summaries = []
        for idx in range(skip_ratios.len):
            skip_summaries.append(1 - skip_ratios.avg[idx])
        cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
        img_size = self.train_dataloader.dataset[0][0].size()[1]
        batch_size = self.train_dataloader.batch_size
        total_flops = get_total_flops(self.pruner.model, img_size)
        gpu_inference_time = get_gpu_inference_time(self.pruner.original_model, img_size, batch_size)
        total_parameters = self.pruner.get_nb_parameters()
        total_parameters_per_module = self.pruner.get_nb_parameters_per_module()
        stat = {
            'Indicator/total_flops': total_flops,
            'Indicator/total_parameters': total_parameters,
            'Indicator/total_parameters_per_module': total_parameters_per_module,
            'Indicator/gpu_inference_time': gpu_inference_time,
            'Indicator/prune_ratio': 0,
            'Indicator/prune_ratio_per_module': {name: 0 for name in total_parameters_per_module.keys()},
            'Performance/train_loss': train_loss,
            'Performance/train_acc': train_acc,
            'Performance/val_loss': val_loss,
            'Performance/val_acc': val_acc,
        }
        if self.pruner.skip:
            stat.update({'Indicator/Computation_percentage': cp.item()})
        if self.steps != 0:
            initial_nb_parameters = self.stats[0]['Indicator/total_parameters']
            initial_nb_parameters_per_module = self.stats[0]['Indicator/total_parameters_per_module']
            stat['Indicator/prune_ratio'] = (initial_nb_parameters - total_parameters)/initial_nb_parameters
            ratio_per_module = {}
            for name in total_parameters_per_module.keys():
                ratio_per_module[name] =\
                    (initial_nb_parameters_per_module[name] - total_parameters_per_module[name])/\
                    initial_nb_parameters_per_module[name]
            stat['Indicator/prune_ratio_per_module'] = ratio_per_module
        enable_print()
        return stat

    def _add_extra_stats(self):
        self.stats[self.steps].update(self.pruner.extra_stat)

    def _update_stats(self):
        stat = self._compute_basic_stats()
        self.stats[self.steps] = stat
        if 0 < len(self.pruner.extra_stat):
            self._add_extra_stats()

    def _update_writer(self):
        for name in self.stats[self.steps].keys():
            if isinstance(self.stats[self.steps][name], dict):
                self.writer.add_scalars(name, self.stats[self.steps][name], self.prune_steps)
            elif isinstance(self.stats[self.steps][name], list):
                self.writer.add_histogram(name, np.array(self.stats[self.steps][name]), self.prune_steps)
            else:
                self.writer.add_scalar(name, self.stats[self.steps][name], self.prune_steps)

    def _save_stats(self):
        with open(os.path.join(self.stats_dir, 'stats_{}.json'.format(self.file_name)), 'w') as f:
            json.dump(self.stats, f, cls=NumpyEncoder)

    def _save_model(self):
        prune_ratio = self.stats[self.steps]['Indicator/prune_ratio']
        val_acc = self.stats[self.steps]['Performance/val_acc']
        filename = 'checkpoint_{}_{:.4f}_{:.4f}.pth'.format(self.file_name, val_acc, prune_ratio)
        torch.save(self.pruner.model.state_dict(), '{}/{}'.format(self.checkpoint_dir, filename))

    def init_steps(self):
        self.steps = 0
        self.prune_steps = 0

    def update_steps(self):
        self.steps += 1
        if hasattr(self.pruner, 'pool'):
            if self.pruner.pool.it % 2 == 0:
                self.prune_steps += 1
        else:
            self.prune_steps += 1

    def add_results(self):
        self._update_stats()
        if hasattr(self.pruner, 'pool'):
            if self.pruner.pool.it % 2 == 1:
                self._update_writer()
        else:
            self._update_writer()
        self._save_stats()
        self._save_model()
        self.update_steps()

    def clean_up(self):
        self.pruner.extra_stat = {}
        self.stats = {}
        self.init_steps()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

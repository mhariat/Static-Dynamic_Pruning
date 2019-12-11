from models.reskipnet import *
from models.pyramid_skipnet import *
from utils.results_utils import *
from prune.low_rank_pruning.methods.improved_eigen_damage import *
from prune.low_rank_pruning.methods.improved_eigen_damage_multi import *
from utils.data_utils import *
from torch import optim
import argparse


pruner_id = {
    0: 'Improved-Eigen-Damage',
    1: 'Improved-Eigen-Damage_multi',
}

scheduler_id = {
    0: 'Exponential',
    1: 'Stair',
    2: 'UpDown'
}


def init_network(config, num_classes):
    if config.network == 'resnet':
        kwargs = {'depth': config.depth, 'num_classes': num_classes}
        depth = config.depth
        net = reskipnet(**kwargs)
    else:
        raise NotImplementedError
    dataset = config.data_dir.split('/')[-1]
    path_to_add = '{}/{}/{}'.format(dataset, config.network, depth)
    assert os.path.exists('{}/checkpoint/sp/{}'.format(config.result_dir, path_to_add)),\
        'No checkpoint directory for sp model!'
    path_to_add = '{}/{}/{}'.format(dataset, config.network, config.depth)
    checkpoint_path, exp_name, epochs = get_best_checkpoint('{}/checkpoint/original/{}'.
                                                            format(config.result_dir, path_to_add))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        net.cuda()
    return net, exp_name, epochs


def init_pruner(config, network):
    kwargs = {'model': network, 'prune_ratio_limit': config.prune_ratio_limit, 'normalize': config.normalize,
              'log_interval': config.log_interval}
    if config.pruner_id in [0, 1]:
        if 'use_bias' in config:
            kwargs.update({'use_bias': config.use_bias})
        if 'allow_increase' in config:
            kwargs.update({'allow_increase': config.allow_increase})
        if 'regularization' in config:
            kwargs.update({'regularization': config.regularization})
        if 'ma' in config:
            kwargs.update({'ma': config.ma})
        if 'fisher_type' in config:
            kwargs.update({'fisher_type': config.fisher_type})
        if 'correct_eigenvalues' in config:
            kwargs.update({'correct_eigenvalues': config.correct_eigenvalues})
        if 'sua' in config:
            kwargs.update({'sua': config.sua})
        if 'use_full_cov_a' in config:
            kwargs.update({'use_full_cov_a': config.use_full_cov_a})
        if 'decomp_method' in config:
            kwargs.update({'decomp_method': config.decomp_method})
        if 'pool_momentum' in config:
            kwargs.update({'pool_momentum': config.pool_momentum})
        if 'back_ratio' in config:
            kwargs.update({'back_ratio': config.back_ratio})
        if config.pruner_id == 0:
            pruner = ImprovedEigenPruner(**kwargs)
        else:
            pruner = ImprovedEigenPrunerMulti(**kwargs)
    else:
        raise NotImplementedError
    dataset = config.data_dir.split('/')[-1]
    pruner.dataset = dataset
    pruner.name = pruner_id[config.pruner_id]
    return pruner


def init_scheduler(config):
    if config.scheduler_id == 0:
        return ExpLRScheduler()
    elif config.scheduler_id == 1:
        return StairLRScheduler()
    elif config.scheduler_id == 2:
        return UpDownLRScheduler()


def main(config):
    print(config)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device_used = 'GPU'
    else:
        device_used = 'CPU'
    train_dataloader, val_dataloader = get_dataloader(data_dir=config.data_dir, resize=config.resize,
                                                      img_type=config.img_type, batch_size=config.batch_size,
                                                      num_workers=config.num_workers)

    num_classes = len(train_dataloader.dataset.classes)
    net, exp_name, original_epochs, _ = init_network(config, num_classes)
    scheduler = init_scheduler(config)
    pruner = init_pruner(config, net)
    if 'exp_name' in config:
        exp_name_results = config.exp_name
    else:
        exp_name_results = None
    result = PruneResults(config.result_dir, train_dataloader, val_dataloader, pruner, exp_name_results)
    logger = result.logger
    scheduler_params = {float(x): y for x, y in config.scheduler_params.items()}
    lr, momentum, weight_decay = config.lr, config.momentum, config.weight_decay
    normalize, prune_ratio, prune_ratio_limit = config.normalize, config.prune_ratio, config.prune_ratio_limit
    epochs_start, epochs_end = config.epochs_start, config.epochs_end
    original_accuracy = result.stats[0]['Performance/val_acc']
    pruner.saved_model.append((original_accuracy, pruner.original_model))
    dataset = config.data_dir.split('/')[-1]
    message = 'Pruning method used: {}. Scheduler used: {}. Device used: {}. Dataset: {}. Number of classes: {}.' \
              ' Network to prune: {}_{}. Original accuracy: {:.2f}% (Exp_name: {}. Trained for {} epochs)'.\
        format(pruner_id[config.pruner_id], scheduler_id[config.scheduler_id], device_used, dataset, num_classes,
               config.network, config.depth, 100*original_accuracy, exp_name, original_epochs)
    print(colored('\n{}\n'.format(message), 'magenta'))
    logger.info(message)
    it = 0
    stop_pruning = False

    while not stop_pruning:
        message = '-' * 200
        print(message)
        print(message)
        logger.info('-' * 150)
        message = '[Pruning method: {}. Iteration: {}]. Pruning. Prune-ratio: {}. Prune-ratio limit: {}'.\
            format(pruner_id[config.pruner_id], it, prune_ratio, prune_ratio_limit)
        print(colored(message, 'magenta'))
        logger.info(message)
        if hasattr(pruner, 'pool'):
            pool_it = pruner.pool.it
        else:
            pool_it = 0
        pruner.prune(prune_ratio=prune_ratio, train_dataloader=train_dataloader)
        initial_nb_parameters = result.stats[0]['Indicator/total_parameters']
        current_compression = (initial_nb_parameters - pruner.get_nb_parameters())/initial_nb_parameters
        epochs = int(ratio_interpolation(start_value=epochs_start, end_value=epochs_end, ratio=current_compression))

        if pool_it % 2 == 1:
            lr_schedule = {50: 0.1, 100: 0.01, 150: 0.001}
            if config.interpolation:
                lr = ratio_interpolation(start_value=10*config.lr, end_value=config.lr, ratio=current_compression)
            else:
                lr = 10*config.lr
        else:
            lr_schedule = {int(x*epochs): y for x, y in scheduler_params.items()}
            lr = config.lr
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        starting_lr = optimizer.param_groups[0]['lr']
        scheduler.settle(lr_schedule=lr_schedule, optimizer=optimizer)
        pruner.fine_tune(epochs=epochs, optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader,
                         val_dataloader=val_dataloader, alpha=config.alpha)
        result.add_results()
        new_compression = result.stats[it + 1]['Indicator/prune_ratio']
        delta_compression = new_compression - current_compression
        if hasattr(pruner, 'allow_increase'):
            if (0 < delta_compression) & (delta_compression < 0.01):
                pruner.allow_increase = True
        current_compression = new_compression
        current_accuracy = result.stats[it + 1]['Performance/val_acc']
        message = '[Pruning method: {}. Iteration: {}]. Accuracy: {:.2f}% [Original Accuracy: {:.2f}%].' \
                  ' Cumulative Compression: {:.2f}%'.format(pruner_id[config.pruner_id], it, 100*current_accuracy,
                                                            100*original_accuracy, 100*current_compression)
        cp = result.stats[it + 1]['Indicator/Computation_percentage']
        message = '{}. Cumulative percentage: {:.3f}.'.format(message, cp)
        print(colored(message, 'green'))
        logger.info(message)
        torch.cuda.empty_cache()
        gc.collect()
        stop_pruning = config.max_iter <= it
        it += 1
    result.clean_up()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to the config json file')
    args = parser.parse_args()
    config_path = args.config_path
    config = get_config_from_json(config_path)
    main(config)

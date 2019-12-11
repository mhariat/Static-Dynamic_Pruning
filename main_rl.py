from models.reskipnet import *
from utils.data_utils import *
from torch import optim
from utils.log import *
from torch.utils.tensorboard import SummaryWriter
import argparse


def init_network(args, num_classes):
    if args.network == 'resnet':
        kwargs = {'depth': args.depth, 'num_classes': num_classes}
        depth = str(args.depth)
        net = reskipnet(**kwargs)
    else:
        raise NotImplementedError
    net.rl_mode = True
    dataset = args.data_dir.split('/')[-1]
    path_to_add = '{}/{}/{}'.format(dataset, args.network, depth)
    assert os.path.exists('{}/checkpoint/sp/{}'.format(args.result_dir, path_to_add)),\
        'No checkpoint directory for sp model!'
    checkpoint_path, _, epochs = get_best_checkpoint('{}/checkpoint/sp/{}'.format(args.result_dir, path_to_add))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        net.cuda()
    net.fill_flop_weights()
    return net, epochs, depth


def init_scheduler(id_):
    if id_ == 0:
        return ExpLRScheduler()
    elif id_ == 1:
        return StairLRScheduler()
    elif id_ == 2:
        return UpDownLRScheduler()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, help='Network to be trained')
    parser.add_argument('--depth', type=int, help='Depth of the netowrk')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--alpha', type=float, default=0.1, help='Rewards magnitude.')
    parser.add_argument('--gamma', type=float, default=1, help='Rewards magnitude.')
    parser.add_argument('--rl_weight', type=float, default=0.01, help='Rewards magnitude.')
    parser.add_argument('--img_type', type=str, help='Image type')
    parser.add_argument('--resize', type=int, help='Resize')
    parser.add_argument('--result_dir', type=str, help='Result directory')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--exp_name', type=str, default=None)

    args = parser.parse_args()
    print(args)
    train_dataloader, val_dataloader = get_dataloader(data_dir=args.data_dir, resize=args.resize,
                                                      img_type=args.img_type, batch_size=args.bs,
                                                      num_workers=args.num_workers)
    num_classes = len(train_dataloader.dataset.classes)
    net, sp_epochs, depth = init_network(args, num_classes)

    sp_accuracy, _, skip_ratios = validation_rl(model=net, use_cuda=torch.cuda.is_available(),
                                                val_dataloader=val_dataloader)
    skip_summaries = []
    for idx in range(skip_ratios.len):
        skip_summaries.append((1 - skip_ratios.avg[idx])*net.flop_weights[idx])
    cp = sum(skip_summaries) * 100
    original_cp = cp.item()

    create_dir(args.result_dir)
    dataset = args.data_dir.split('/')[-1]
    path_to_add = '{}/{}/{}'.format(dataset, args.network, depth)
    writer_dir = '{}/writer/rl/{}'.format(args.result_dir, path_to_add)
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = 'unknown'
    run = get_run(writer_dir, exp_name)
    exp_name = 'run_{}_{}'.format(run, exp_name)
    writer_dir = '{}/{}'.format(writer_dir, exp_name)
    checkpoint_dir = '{}/checkpoint/rl/{}'.format(args.result_dir, path_to_add)
    log_dir = '{}/logs/rl/{}'.format(args.result_dir, path_to_add)
    create_dir(writer_dir)
    create_dir(checkpoint_dir)
    writer = SummaryWriter(writer_dir)
    logger = get_logger('log_{}'.format(exp_name), log_dir)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device_used = 'GPU'
    else:
        device_used = 'CPU'
    lr, momentum, weight_decay, epochs = args.lr, args.momentum, args.weight_decay, args.epochs
    message = '-'*200
    print(message)
    print(message)
    message = 'Dataset: {}. Number of classes: {}. Network: {}_{}. Number of Epochs: {}. Device used: {}.'.\
        format(dataset, num_classes, args.network, depth, args.epochs, device_used)
    message = '{}. RL part. SP accuracy: {:.2f}. SP epochs: {}. SP Computation Percentage: {:.2f}'.\
        format(message, 100*sp_accuracy, sp_epochs, original_cp)
    print(colored(message, 'magenta'))
    logger.info(message)
    logger.info('-'*150)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    starting_lr = optimizer.param_groups[0]['lr']
    lr = starting_lr
    total_rewards = AverageMeter()
    best_acc = 0

    for epoch in range(epochs):
        message = '-' * 200
        print(message)
        print(message)
        train_acc, train_loss = train_rl(net, optimizer, use_cuda, train_dataloader, epoch, total_rewards,
                                         args.alpha, args.gamma, args.rl_weight, args.log_interval)
        val_acc, val_loss, skip_ratios = validation_rl(model=net, use_cuda=use_cuda, val_dataloader=val_dataloader)
        skip_summaries = []
        for idx in range(skip_ratios.len):
            skip_summaries.append((1 - skip_ratios.avg[idx])*net.flop_weights[idx])
        # cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
        cp = sum(skip_summaries) * 100
        message = 'Training. Epoch: [{}/{}]. Train Loss: {:.6f}. Train Accuracy: {:.2f}%. Learning rate {:.5f}.' \
                  ' Validation loss: {:.6f}. Validation Accuracy: {}/{} ({:.2f}%). Computation Percentage: {:.3f} %'.\
            format(epoch, epochs, train_loss, 100 * train_acc, lr, val_loss, int(100*val_acc),
                   len(val_dataloader.dataset), 100. * val_acc, cp)
        print(colored('\n{}\n'.format(message), 'yellow'))
        logger.info(message)
        if best_acc < val_acc:
            filename = 'checkpoint_{}_{:.4f}_{:.4f}_{:.0f}.pth'.format(exp_name, val_acc, cp, epoch)
            torch.save(net.state_dict(), '{}/{}'.format(checkpoint_dir, filename))
        stat = {
            'Performance/train_loss': train_loss,
            'Performance/train_acc': train_acc,
            'Performance/val_loss': val_loss,
            'Performance/val_acc': val_acc,
            'Indicator/computation_percentage': cp
        }
        for name in stat.keys():
            writer.add_scalar(name, stat[name], epoch)


if __name__ == '__main__':
    main()

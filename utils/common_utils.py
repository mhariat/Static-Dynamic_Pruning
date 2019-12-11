import os
import sys
import torch.nn as nn
import torch
import gc
from termcolor import colored
import numpy as np
from abc import abstractmethod, ABCMeta
from torch.optim import lr_scheduler
import json
from easydict import EasyDict
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from torchnet.engine import Engine


def test(model, testloader, loss_function, device):
    model.eval()
    model.to(device)

    engine = Engine()

    def compute_loss(data):
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        return loss_function(outputs, labels), outputs

    def on_start(state):
        print("Running inference ...")
        state['iterator'] = tqdm(state['iterator'], leave=False)

    class Accuracy():
        _accuracy = 0.
        _sample_size = 0.

    def on_forward(state):
        batch_size = state['sample'][1].shape[0]
        Accuracy._sample_size += batch_size
        Accuracy._accuracy += batch_size * get_accuracy(state['output'].cpu(), state['sample'][1].cpu())

    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward

    engine.test(compute_loss, testloader)

    return Accuracy._accuracy / Accuracy._sample_size


def get_accuracy(outputs, labels):
    __, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()
    return accuracy


class BatchCrossEntropy(nn.Module):
    def __init__(self):
        super(BatchCrossEntropy, self).__init__()

    def forward(self, x, target):
        logp = F.log_softmax(x)
        target = target.view(-1,1)
        output = - logp.gather(1, target)
        return output


def expand_model(model):
    layers = []
    for layer in model.children():
        if len(list(layer.children())) > 0:
            layers += expand_model(layer)
        else:
            layers.append(layer)
    return layers


def count_nonzero_parameters(module):
    return torch.gt(torch.abs(module.weight.data), 0).sum().item()


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def train_sp(model, optimizer, use_cuda, train_dataloader, epoch, alpha=1e-5, log_interval=100, multi_gpu=False):

    if multi_gpu:
        model = nn.DataParallel(model.cuda())
    model.train()
    iteration = 0
    train_loss = 0
    correct = 0
    nb_iterations = int(len(train_dataloader.dataset)/train_dataloader.batch_size)

    for batch_idx, (data, target) in enumerate(train_dataloader):

        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, masks, _ = model(data)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        sparsity_loss = 0
        for mask in masks:
            sparsity_loss += mask.mean()
        loss = criterion(output, target) + alpha*sparsity_loss
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct_batch = pred.eq(target.data.view_as(pred)).sum().item()
        correct += correct_batch
        train_loss = loss.data.item()
        train_accuracy = correct_batch/train_dataloader.batch_size
        if iteration % log_interval == 0:
            print('\nTrain Epoch: {}. Iteration: [{:.0f}/{:.0f}]. Loss: {:.6f}. Accuracy: {:.2f}%.\n'.
                  format(epoch, iteration, nb_iterations, train_loss, 100 * train_accuracy))
        iteration += 1
    train_accuracy = correct/len(train_dataloader.dataset)
    train_loss /= len(train_dataloader.dataset)
    torch.cuda.empty_cache()
    gc.collect()
    if multi_gpu:
        model = model.module
    return train_accuracy, train_loss


def validation_sp(model, use_cuda, val_dataloader):
    skip_ratios = ListAverageMeter()
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output, masks, _ = model(data)
            skips = [mask.data.le(0.5).float().mean() for mask in masks]
            if skip_ratios.len != len(skips):
                skip_ratios.set_len(len(skips))
            skip_ratios.update(skips, data.size(0))
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    torch.cuda.empty_cache()
    gc.collect()
    validation_loss /= len(val_dataloader.dataset)
    validation_accuracy = correct/len(val_dataloader.dataset)
    return validation_accuracy, validation_loss, skip_ratios


def train_rl(model, optimizer, use_cuda, train_dataloader, epoch, total_rewards, alpha, gamma, rl_weight,
             log_interval=100, multi_gpu=False):
    gate_saved_actions = model.control.saved_actions
    gate_rewards = model.control.rewards

    batch_criterion = BatchCrossEntropy()
    criterion = nn.CrossEntropyLoss()
    if multi_gpu:
        model = nn.DataParallel(model.cuda())
    model.train()
    iteration = 0
    train_loss = 0
    correct = 0
    nb_iterations = int(len(train_dataloader.dataset)/train_dataloader.batch_size)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output, masks, probs = model(data)

        pred_loss = batch_criterion(output, target)
        # normalized_alpha = alpha / len(gate_saved_actions)
        normalized_alpha = [alpha*w for w in model.flop_weights]
        ct = 0
        for act in gate_saved_actions:
            gate_rewards.append((1 - act.float()).data * normalized_alpha[ct])
            ct += 1
        R = - pred_loss.data
        cum_rewards = []
        for r in gate_rewards[::-1]:
            R = r + gamma * R
            cum_rewards.insert(0, R)
        ct = 0
        for action, prob, R in zip(gate_saved_actions, probs, cum_rewards):
            dist = Categorical(prob)
            _loss = -dist.log_prob(action)*R
            if ct == 0:
                rl_losses = _loss
            else:
                rl_losses = rl_losses + _loss
            ct += 1
        rl_losses = rl_losses.mean()

        loss = criterion(output, target) + rl_weight*rl_losses
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct_batch = pred.eq(target.data.view_as(pred)).sum().item()
        correct += correct_batch
        train_loss = loss.data.item()
        train_accuracy = correct_batch/train_dataloader.batch_size
        total_rewards.update(cum_rewards[0].mean(), data.size(0))
        total_gate_reward = sum([r.mean() for r in gate_rewards])
        if iteration % log_interval == 0:
            message = 'Train Epoch: {}. Iteration: [{:.0f}/{:.0f}]. Loss: {:.6f}. Accuracy: {:.2f}%.' \
                      ' Total reward {total_rewards.val: .3f} ({total_rewards.avg: .3f}).' \
                      ' Total gate reward {total_gate_reward: .3f}'.\
                format(epoch, iteration, nb_iterations, train_loss, 100 * train_accuracy, total_rewards=total_rewards,
                       total_gate_reward=total_gate_reward)
            print('\n{}\n'.format(message))
        iteration += 1
        del gate_saved_actions[:]
        del gate_rewards[:]

    train_accuracy = correct/len(train_dataloader.dataset)
    train_loss /= len(train_dataloader.dataset)
    torch.cuda.empty_cache()
    gc.collect()
    if multi_gpu:
        model = model.module
    return train_accuracy, train_loss


def validation_rl(model, use_cuda, val_dataloader):
    gate_saved_actions = model.control.saved_actions
    gate_rewards = model.control.rewards
    skip_ratios = ListAverageMeter()
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output, masks, _ = model(data)
            skips = [mask.data.le(0.5).float().mean() for mask in masks]
            if skip_ratios.len != len(skips):
                skip_ratios.set_len(len(skips))
            skip_ratios.update(skips, data.size(0))
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    torch.cuda.empty_cache()
    gc.collect()
    validation_loss /= len(val_dataloader.dataset)
    validation_accuracy = correct/len(val_dataloader.dataset)
    del gate_saved_actions[:]
    del gate_rewards[:]
    return validation_accuracy, validation_loss, skip_ratios


def display_channels(model):
    res = []
    all_layer = expand_model(model)
    for k in range(len(all_layer)):
        layer = all_layer[k]
        if isinstance(layer, nn.Conv2d) | isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Conv2d):
                res.append(('Conv2d_{}'.format(k), [layer.in_channels, layer.out_channels]))
            else:
                res.append(('Fc_{}'.format(k), [layer.in_features, layer.out_features]))
    print(colored(res, 'red'))


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    return EasyDict(config_dict)


def ratio_interpolation(start_value, end_value, ratio):
    return (1 - ratio)*start_value + ratio*end_value


def get_best_checkpoint(dir, n=2):
    """file format should be: checkpoint_[Exp_name]_[Accuracy]_[Epochs]"""
    files = os.listdir(dir)
    assert 0 < len(files), 'No checkpoint for this network !'
    best_id = int(np.argmax([float(file.split('_')[-n]) for file in files]))
    exp_name = files[best_id].split('_')[3]
    epochs = files[best_id].split('_')[-1].split('.')[0]
    return '{}/{}'.format(dir, files[best_id]), exp_name, int(epochs)


def get_run(dir, pattern):
    if os.path.exists(dir):
        run_numbers = [int(file.split('_')[1]) for file in os.listdir(dir) if pattern in file]
        if len(run_numbers) == 0:
            return 0
        else:
            return max(run_numbers) + 1
    else:
        return 0


class LRScheduler(metaclass=ABCMeta):
    def __init__(self):
        self.optimizer = None
        self.scheduler = None
        self.lr_schedule = None

    @abstractmethod
    def lambda_rule(self, epoch):
        pass

    def settle(self, lr_schedule, optimizer):
        self.lr_schedule = {0: 1}
        self.lr_schedule.update(lr_schedule)
        self.optimizer = optimizer
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lambda_rule)

    def update_lr(self, epoch):
        self.scheduler.step(epoch)
        lr = self.optimizer.param_groups[0]['lr']
        return lr


class ExpLRScheduler(LRScheduler):

    def lambda_rule(self, epoch):
        key_epochs = list(self.lr_schedule.keys())
        key_lrs = list(self.lr_schedule.values())
        for k in range(len(key_epochs)-1):
            if (key_epochs[k] <= epoch) & (epoch < key_epochs[k+1]):
                rate = key_lrs[k+1]/key_lrs[k]
                x = (epoch - key_epochs[k]) / (key_epochs[k+1] - key_epochs[k])
                gamma = -np.log(rate)
                return key_lrs[k]*np.exp(-gamma * x)
        return key_lrs[-1]


class StairLRScheduler(LRScheduler):

    def lambda_rule(self, epoch):
        key_epochs = list(self.lr_schedule.keys())
        key_lrs = list(self.lr_schedule.values())
        for k in range(len(key_epochs)-1):
            if (key_epochs[k] <= epoch) & (epoch < key_epochs[k+1]):
                return key_lrs[k]
        return key_lrs[-1]


class UpDownLRScheduler(LRScheduler):

    def settle(self, lr_schedule, optimizer):
        assert len(lr_schedule) == 2, "Schedule dictionary should be of lenght 2 for this scheduler."
        super(UpDownLRScheduler, self).settle(lr_schedule, optimizer)

    def lambda_rule(self, epoch):
        self.stops = list(self.lr_schedule.keys())
        self.lrs = list(self.lr_schedule.values())
        if epoch < self.stops[0]:
            x = epoch/self.stops[0]
            gamma = np.log(self.lrs[0])
            return np.exp(x*gamma)
        elif (self.stops[0] <= epoch) & (epoch < self.stops[1]):
            x = (epoch - self.stops[0])/(self.stops[1] - self.stops[0])
            gamma = -np.log(self.lrs[1]/self.lrs[0])
            return self.lrs[0]*np.exp(-x*gamma)
        else:
            return self.lrs[1]


def moving_average(prev, new, momentum):
    decay = momentum/(1 - momentum)
    prev *= decay
    prev += new
    prev *= (1-momentum)


def get_new_module(W, prev_module):
    prev_cout = prev_module.weight.data.size(0)
    if isinstance(prev_module, nn.Conv2d):
        cout, cin, kh, kw = W.shape
        if prev_module.bias is not None:
            new_module = nn.Conv2d(out_channels=cout, in_channels=cin, kernel_size=(kh, kw), stride=prev_module.stride,
                                   padding=prev_module.padding, bias=True)
            last = prev_module.bias.data[-1]
            new_module.bias.data = prev_module.bias.data.new_zeros(cout)
            new_module.bias.data[:prev_cout] = prev_module.bias.data[:prev_cout]
            new_module.bias.data[prev_cout:] = last
        else:
            new_module = nn.Conv2d(out_channels=cout, in_channels=cin, kernel_size=(kh, kw), stride=prev_module.stride,
                                   padding=prev_module.padding, bias=False)
        new_module.weight.data = W
    elif isinstance(prev_module, nn.Linear):
        cout, cin = W.shape
        if prev_module.bias is not None:
            new_module = nn.Linear(out_features=cout, in_features=cin, bias=True)
            new_module.bias.data = prev_module.bias.data.new_zeros(cout)
            new_module.bias.data[:prev_cout] = prev_module.bias.data[:prev_cout]
            last = prev_module.bias.data[-1]
            new_module.bias.data[prev_cout:] = last
        else:
            new_module = nn.Linear(out_features=cout, in_features=cin, bias=False)
        new_module.weight.data = W
    else:
        raise NotImplementedError
    return new_module


def tensor_norm(A):
    return torch.sqrt(torch.sum(A**2)).item()


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                   type(obj.data).__name__,
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "",
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

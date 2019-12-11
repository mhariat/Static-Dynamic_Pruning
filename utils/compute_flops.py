import torch.nn as nn
from collections import OrderedDict
import gc
import torch


def compute_conv2d_flops(module, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()

    non_zeros_parameters = torch.gt(torch.abs(module.weight.data), 0).sum().item()
    all_parameters = module.weight.data.numel()
    sparsity = 1 - non_zeros_parameters/all_parameters

    vector_length = module.kernel_size[0] * module.kernel_size[1] * input_channels * (1 - sparsity)/module.groups
    n_output_elements = output_width * output_height * output_channels

    bias_ops = 1 if module.bias is not None else 0

    flops_mults = vector_length * n_output_elements * batch_size
    flops_adds = (vector_length - 1 + bias_ops) * n_output_elements * batch_size

    return [flops_mults, flops_adds]


def compute_linear_flops(module, input, output):
    batch_size, _ = input[0].size()

    non_zeros_parameters = torch.gt(torch.abs(module.weight.data), 0).sum().item()
    all_parameters = module.weight.data.numel()
    sparsity = 1 - non_zeros_parameters/all_parameters

    cout, cin = module.weight.size()
    n_elements = cin * (1 - sparsity)

    bias_ops = 1 if module.bias is not None else 0

    flop_mults = n_elements * cout * batch_size
    flop_adds = (n_elements - 1 + bias_ops) * cout * batch_size

    return [flop_mults, flop_adds]


def compute_batchnorm2d_flops(module, input, output):
    count = input[0].nelement()
    flop_mults = 1 * count
    flop_adds = 1 * count
    return [flop_mults, flop_adds]


def compute_relu_flops(module, input, output):
    flop_mults = input[0].nelement()
    flop_adds = 0
    return [flop_mults, flop_adds]


def compute_maxpool2d_flops(module, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()

    vector_length = module.kernel_size[0] * module.kernel_size[1]
    n_output_elements = output_width * output_height * output_channels
    flop_adds = 0
    flop_mults = vector_length * n_output_elements * batch_size
    return [flop_mults, flop_adds]


def compute_avgpool2d_flops(module, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()

    if isinstance(module.kernel_size, tuple):
        vector_length = module.kernel_size[0] * module.kernel_size[1]
    else:
        vector_length = module.kernel_size * module.kernel_size
    n_output_elements = output_width * output_height * output_channels
    flop_adds = (vector_length - 1) * n_output_elements * batch_size
    flop_mults = n_output_elements * batch_size
    return [flop_mults, flop_adds]


def compute_softmax_flops(module, input, output):
    count = input[0].nelement()
    flop_mults = 2 * count
    flop_adds = 1 * count
    return [flop_mults, flop_adds]


def compute_sigmoid_flops(module, input, output):
    count = input[0].nelement()
    flop_mults = 2 * count
    flop_adds = 1 * count
    return [flop_mults, flop_adds]


def compute_lstm_flops(module, input, output):
    batch_size = input[0].size(0)
    embedding_size = module.input_size
    hidden_size = module.hidden_size

    weight_ops_mul = embedding_size
    weight_ops_add = weight_ops_mul - 1
    bias_ops = 1
    linear_mult_flops = hidden_size * weight_ops_mul
    linear_add_flops = hidden_size * (weight_ops_add + bias_ops)
    activation_flops = hidden_size

    element_wise_multi_flops = hidden_size
    element_wise_add_flops = hidden_size

    flop_mults = 8*linear_mult_flops + 9*activation_flops + 3*element_wise_multi_flops
    flop_adds = 8*linear_add_flops + element_wise_add_flops

    flop_mults *= batch_size
    flop_adds *= batch_size
    return [flop_mults, flop_adds]


def get_total_flops(model, input_res, test_dataloader=None):
    cuda = torch.cuda.is_available()

    list_conv = []

    def conv_hook(self, input, output):
        res = compute_conv2d_flops(self, input, output)
        list_conv.append(res)

    list_linear = []

    def linear_hook(self, input, output):
        res = compute_linear_flops(self, input, output)
        list_linear.append(res)

    list_bn = []

    def bn_hook(self, input, output):
        res = compute_batchnorm2d_flops(self, input, output)
        list_bn.append(res)

    list_relu = []

    def relu_hook(self, input, output):
        res = compute_relu_flops(self, input, output)
        list_relu.append(res)

    def relu6_hook(self, input, output):
        res = 2*compute_relu_flops(self, input, output)
        list_relu.append(res)

    list_pooling = []

    def max_pooling_hook(self, input, output):
        res = compute_maxpool2d_flops(self, input, output)
        list_pooling.append(res)

    def avg_pooling_hook(self, input, output):
        res = compute_avgpool2d_flops(self, input, output)
        list_pooling.append(res)

    def softmax_pooling_hook(self, input, output):
        res = compute_softmax_flops(self, input, output)
        list_pooling.append(res)

    def sigmoid_pooling_hook(self, input, output):
        res = compute_sigmoid_flops(self, input, output)
        list_pooling.append(res)

    list_lstm = []

    def lstm_hook(self, input, output):
        res = compute_lstm_flops(self, input, output)
        list_lstm.append(res)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.ReLU6):
                net.register_forward_hook(relu6_hook)
            if isinstance(net, torch.nn.MaxPool2d):
                net.register_forward_hook(max_pooling_hook)
            if isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(avg_pooling_hook)
            if isinstance(net, torch.nn.Softmax):
                net.register_forward_hook(softmax_pooling_hook)
            if isinstance(net, torch.nn.Sigmoid):
                net.register_forward_hook(sigmoid_pooling_hook)
            if isinstance(net, torch.nn.LSTM):
                net.register_forward_hook(lstm_hook)
            return
        for c in childrens:
            foo(c)

    def _rm_hooks(model):
        for m in model.modules():
            m._forward_hooks = OrderedDict()

    foo(model)

    if test_dataloader:
        with torch.no_grad():
            for data, target in test_dataloader:
                if cuda:
                    data, target = data.cuda(), target.cuda()
                out = model(data)
    else:
        if cuda:
            input = torch.cuda.FloatTensor(1, 3, input_res, input_res)
        else:
            input = torch.FloatTensor(1, 3, input_res, input_res)
        with torch.no_grad():
            out = model(input)

    torch.cuda.empty_cache()
    list_total = list_conv + list_linear + list_bn + list_relu + list_pooling + list_lstm

    total_flop_mults = sum(t[0] for t in list_total)
    total_flop_adds = sum(t[1] for t in list_total)

    residual_addition_flops = 0
    for module in model.modules():
        if hasattr(module, 'additional_flops'):
            residual_addition_flops += module.additional_flops
    if 1 < len(out):
        classification_flops = out[0].nelement() - 1
    else:
        classification_flops = out.nelement() - 1

    _rm_hooks(model)
    torch.cuda.empty_cache()
    gc.collect()

    total_flop_mults += classification_flops
    total_flop_adds += residual_addition_flops

    if test_dataloader:
        test_dataloader_size = len(test_dataloader.dataset)
        total_flop_mults /= test_dataloader_size
        total_flop_adds /= test_dataloader_size
    total_flops = total_flop_adds + total_flop_mults
    return total_flops


import torch
import time
import gc


def get_gpu_inference_time(net, input_res, batch_size):
    x = torch.cuda.FloatTensor(batch_size, 3, input_res, input_res).normal_()
    net = net.cuda()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    a = time.perf_counter()
    with torch.no_grad():
        out = net(x)
    torch.cuda.synchronize()
    b = time.perf_counter()
    torch.cuda.empty_cache()
    gc.collect()
    net = net.cpu()
    return (b - a)/batch_size


def get_cpu_inference_time(net, input_res, batch_size):
    x = torch.FloatTensor(batch_size, 3, input_res, input_res).normal_()
    start_time = time.time()
    with torch.no_grad():
        out = net(x)
    end_time = time.time()
    return (end_time - start_time)/batch_size

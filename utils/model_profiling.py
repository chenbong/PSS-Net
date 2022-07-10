import numpy as np
import time
import torch
import torch.nn as nn

from .comm import set_active_subnet
from models.slimmable_ops import USBatchNorm2d

model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 95
params_space = 15
macs_space = 15
seconds_space = 15

num_forwards = 10


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.time = self.end - self.start
        if self.verbose:
            print('Elapsed time: %f ms.' % self.time)


def get_params(self):
    """get number of params in module"""
    return np.sum(
        [np.prod(list(w.size())) for w in self.parameters()])


def run_forward(self, inputs):
    with Timer() as t:
        for _ in range(num_forwards):
            self.forward(*inputs)
            torch.cuda.synchronize()
    return int(t.time * 1e9 / num_forwards)

def conv_name(cin, cout, k, s, p, g, b):
    return f'Conv2d({cin:d}, {cout:d}, k={k}, s={s}, pad={p}, groups={g:d}, b={b})'

def module_profiling(self, inputs, output, verbose):
    ins = inputs[0].size()
    outs = output.size()
    t = type(self)
    if isinstance(self, nn.Conv2d):
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.f_groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, inputs)
        self.name = conv_name(ins[1], outs[1], self.kernel_size[0], self.stride, self.padding, self.f_groups, self.bias)
    elif isinstance(self, nn.Linear):
        self.n_macs = ins[1] * outs[1] * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, inputs)
        self.name = self.__repr__()
    elif isinstance(self, nn.AvgPool2d):
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = run_forward(self, inputs)
        self.name = self.__repr__()
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = run_forward(self, inputs)
        self.name = self.__repr__()
    else:
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        num_children = 0
        for m in self.children():
            self.n_macs += getattr(m, 'n_macs', 0)
            self.n_params += getattr(m, 'n_params', 0)
            self.n_seconds += getattr(m, 'n_seconds', 0)
            num_children += 1
        ignore_zeros_t = [
            nn.BatchNorm2d, nn.Dropout2d, nn.Dropout, nn.Sequential,
            nn.ReLU6, nn.ReLU, nn.MaxPool2d,
            nn.modules.padding.ZeroPad2d, nn.modules.activation.Sigmoid,
            USBatchNorm2d,
        ]
        if (not getattr(self, 'ignore_model_profiling', False) and
                self.n_macs == 0 and
                t not in ignore_zeros_t):
            print(
                'WARNING: leaf module {} has zero n_macs.'.format(type(self)))
        return
    if verbose:
        print(
            self.name.ljust(name_space, ' ') +
            '{:,}'.format(self.n_params).rjust(params_space, ' ') +
            '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
            '{:,}'.format(self.n_seconds).rjust(seconds_space, ' '))
    return


def add_profiling_hooks(m, verbose):
    global model_profiling_hooks
    model_profiling_hooks.append(
        m.register_forward_hook(lambda m, inputs, output: module_profiling(m, inputs, output, verbose=verbose)))


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(model, model_cfg, batch=1, channel=3, resolution=224, use_cuda=False, print_=False, verbose=False):
    """ Pytorch model profiling with inputs image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).

    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool

    Returns:
        macs: int
        params: int

    """
    set_active_subnet(model, model_cfg)
    model.eval()
    data = torch.rand(batch, channel, resolution, resolution)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    data = data.to(device)
    model.apply(lambda m: add_profiling_hooks(m, verbose=verbose))
    if print_:
        print(
            'Item'.ljust(name_space, ' ') +
            'params'.rjust(macs_space, ' ') +
            'macs'.rjust(macs_space, ' ') +
            'nanosecs'.rjust(seconds_space, ' '))
    if print_ and verbose:
        print(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
    model(data)
    if print_ and verbose:
        print(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
    if print_:
        print(
            'Total'.ljust(name_space, ' ') +
        '{:.2f}M'.format(model.n_params/1e6).rjust(params_space, ' ') +
        '{:.2f}M'.format(model.n_macs/1e6).rjust(macs_space, ' ') +
        '{:.2f}M'.format(model.n_seconds/1e6).rjust(seconds_space, ' '))
    remove_profiling_hooks()
    model.apply(lambda m: setattr(m, 'profiling', False))
    return model.n_macs, model.n_params

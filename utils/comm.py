import torch
import torch.nn
import pickle
from collections import namedtuple


BlockCfg = namedtuple('BlockCfg', 'block_type in_size cin cout cmid kernel_size stride padding', defaults=[None, None, None, None])

def create_model_block_list(model_cfg):
    block_list = []
    _kernel_size, _stride, _padding = 3, 2, 1
    if model_cfg['model_name'] == 'mobilenet_v1':
        base_stage_couts = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512,   512, 512, 1024, 1024]
        stage_cout_mults = model_cfg['stage_cout_mults']
        stage_couts = [make_divisible(base_stage_couts[i]*cout_mult) for i, cout_mult in enumerate(stage_cout_mults)]

        in_size = model_cfg['resolution']
        block_list += [BlockCfg('Conv2d', in_size,                3, stage_couts[0], kernel_size=3, stride=2, padding=1),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1

        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[0], stage_couts[1], cmid=None, kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[1], stage_couts[2], cmid=None, kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[2], stage_couts[3], cmid=None, kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[3], stage_couts[4], cmid=None, kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[4], stage_couts[5], cmid=None, kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[5], stage_couts[6], cmid=None, kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[6], stage_couts[7], cmid=None, kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[7], stage_couts[8], cmid=None, kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[8], stage_couts[9], cmid=None, kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[9], stage_couts[10], cmid=None, kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[10], stage_couts[11], cmid=None, kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[11], stage_couts[12], cmid=None, kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV1Block', in_size, stage_couts[12], stage_couts[13], cmid=None, kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('Linear',          1, stage_couts[13],          1000, cmid=None, kernel_size=None, stride=None, padding=None),]

    elif model_cfg['model_name'] == 'mobilenet_v2':
        base_stage_couts = [32, 16, 24, 32, 64, 96, 160, 320, 1280]                             # len=9
        base_block_cmids = [16*6]*1 + [24*6]*2 + [32*6]*3 + [64*6]*4 + [96*6]*3 + [160*6]*3     # len=16

        stage_cout_mults = model_cfg['stage_cout_mults']
        stage_couts = [make_divisible(base_stage_couts[i]*cout_mult) for i, cout_mult in enumerate(stage_cout_mults)]
        if 'block_cmid_mults' in model_cfg:
            block_cmid_mults = model_cfg['block_cmid_mults']
            block_cmids = [make_divisible(base_block_cmids[i]*cout_mult) for i, cout_mult in enumerate(block_cmid_mults)]
        else:
            block_cmids = [
                stage_couts[1], stage_couts[2],
                stage_couts[2], stage_couts[3], stage_couts[3],
                stage_couts[3], stage_couts[4], stage_couts[4], stage_couts[4],
                stage_couts[4], stage_couts[5], stage_couts[5], 
                stage_couts[5], stage_couts[6], stage_couts[6], 
                stage_couts[6], 
            ]
            block_cmids = [cmid*6 for cmid in block_cmids]
        
        

        in_size = model_cfg['resolution']

        # head
        block_list += [BlockCfg('Conv2d', in_size,                3, stage_couts[0], kernel_size=3, stride=2, padding=1),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        
        # blocks
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[0], stage_couts[1], cmid=stage_couts[0], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[1], stage_couts[2], cmid=block_cmids[0], kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[2], stage_couts[2], cmid=block_cmids[1], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[2], stage_couts[3], cmid=block_cmids[2], kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[3], stage_couts[3], cmid=block_cmids[3], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[3], stage_couts[3], cmid=block_cmids[4], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[3], stage_couts[4], cmid=block_cmids[5], kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[4], stage_couts[4], cmid=block_cmids[6], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[4], stage_couts[4], cmid=block_cmids[7], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[4], stage_couts[4], cmid=block_cmids[8], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[4], stage_couts[5], cmid=block_cmids[9], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[5], stage_couts[5], cmid=block_cmids[10], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[5], stage_couts[5], cmid=block_cmids[11], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[5], stage_couts[6], cmid=block_cmids[12], kernel_size=None, stride=2, padding=None),]; in_size = int((in_size + _padding * 2 - _kernel_size) / _stride) + 1
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[6], stage_couts[6], cmid=block_cmids[13], kernel_size=None, stride=1, padding=None),]
        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[6], stage_couts[6], cmid=block_cmids[14], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('MBV2Block', in_size, stage_couts[6], stage_couts[7], cmid=block_cmids[15], kernel_size=None, stride=1, padding=None),]

        block_list += [BlockCfg('Conv2d',    in_size, stage_couts[7], stage_couts[8], kernel_size=1, stride=1, padding=0),]

        # classifier
        block_list += [BlockCfg('Linear',          1, stage_couts[8],          1000, cmid=None, kernel_size=None, stride=None, padding=None),]


    else:
        raise NotImplementedError

    return block_list



class LatencyPredictor():
    def __init__(self, lut_dir) -> None:
        with open(lut_dir, 'rb') as f:
            self.lut = pickle.load(f)
    
    def predict_subnet_latency(self, model_cfg, verbose=None):
        subnet_block_list = create_model_block_list(model_cfg)

        total_us = 0.
        for block_cfg in subnet_block_list:
            if block_cfg in self.lut:
                total_us += self.lut[block_cfg]
            else:
                print(subnet_block_list)
                print(block_cfg)
                raise IndexError
        return total_us


def make_divisible(v, divisor=8, min_value=8):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

def round_metric(metric, divisor, offset):
    return int(round((metric - offset) / divisor) * divisor + offset)

def exp(Vs, Ve, e, E):
    return Vs * (Ve/Vs)**(e/E)


def adapt_channels(model_cfg):
    if model_cfg['model_name'] == 'mobilenet_v1':
        stage_cout_mults = model_cfg['stage_cout_mults']
        assert len(stage_cout_mults) == 14

        width_mults = [
                  stage_cout_mults[0],

            None, stage_cout_mults[1],

            None, stage_cout_mults[2],
            None, stage_cout_mults[3],

            None, stage_cout_mults[4],
            None, stage_cout_mults[5],

            None, stage_cout_mults[6],
            None, stage_cout_mults[7],
            None, stage_cout_mults[8],
            None, stage_cout_mults[9],
            None, stage_cout_mults[10],
            None, stage_cout_mults[11],

            None, stage_cout_mults[12],
            None, stage_cout_mults[13],
        ]
        return width_mults

    elif model_cfg['model_name'] == 'mobilenet_v2':
        stage_cout_mults = model_cfg['stage_cout_mults']
        assert len(stage_cout_mults) == 9
        width_mults = [
                        stage_cout_mults[0],


                  None, stage_cout_mults[1],

            None, None, stage_cout_mults[2],
            None, None, stage_cout_mults[2],
            
            None, None, stage_cout_mults[3],
            None, None, stage_cout_mults[3],
            None, None, stage_cout_mults[3],

            None, None, stage_cout_mults[4],
            None, None, stage_cout_mults[4],
            None, None, stage_cout_mults[4],
            None, None, stage_cout_mults[4],

            None, None, stage_cout_mults[5],
            None, None, stage_cout_mults[5],
            None, None, stage_cout_mults[5],

            None, None, stage_cout_mults[6],
            None, None, stage_cout_mults[6],
            None, None, stage_cout_mults[6],

            None, None, stage_cout_mults[7],


                        stage_cout_mults[8],
        ]
        
        

        return width_mults

    else:
        print('NotImplementedError')
        raise NotImplementedError


def _calc_conv2d_flops(input_size, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, verbose=False):
    assert kernel_size % 2 == 1, 'kernel_size should be odd.'
    output_size = int((input_size + padding * 2 - kernel_size) / stride) + 1
    flops = (output_size**2 * (out_channels / groups)) * (kernel_size**2 * in_channels)

    if verbose:
        print(f'Conv2d({in_channels}, {out_channels}, k={kernel_size}, s=({stride}, {stride}), pad=({padding}, {padding}), group={groups}, b={bias})\t\t\t\t{flops:,}')
    return flops, output_size, out_channels


def _calc_avgpool_flops(input_size, in_channels, output_size, verbose=False):
    flops = input_size**2 * in_channels
    out_channels = in_channels
    if verbose:
        print(f'AvgPool2d(output_size=({output_size}, {output_size}))\t\t\t\t{flops:,}')
    return flops, output_size, out_channels


def _calc_fc_flops(in_channels, out_channels, verbose=False):
    flops = in_channels*out_channels
    if verbose:
        print(f'Linear(in={in_channels}, out={out_channels})\t\t\t\t{flops:,}')
    return flops

def _calc_mbv1_block_flops(input_size, cin, cout, stride, cur_layer_id, width_mults, verbose=False):
    f_cin = make_divisible(cin * width_mults[cur_layer_id-1])
    total_flops = 0

    flops, input_size, cin = _calc_conv2d_flops(input_size, f_cin, f_cin, kernel_size=3, stride=stride, padding=1, groups=f_cin, bias=False, verbose=verbose)
    total_flops += flops
    cur_layer_id += 1

    f_cout = make_divisible(cout * width_mults[cur_layer_id])
    flops, input_size, cin = _calc_conv2d_flops(input_size, f_cin, f_cout, kernel_size=1, stride=1, padding=0, bias=False, verbose=verbose)
    total_flops += flops

    return total_flops, input_size, f_cout


def _calc_mbv2_block_flops(input_size, cin, cout, stride, expand_ratio, cur_layer_id, width_mults, verbose=False):
    f_cin = make_divisible(cin * width_mults[cur_layer_id-1])

    total_flops = 0
    if expand_ratio != 1:
        if width_mults[cur_layer_id]:
            f_expand = make_divisible(cin * expand_ratio * width_mults[cur_layer_id])
        else:
            f_expand = make_divisible(f_cin * expand_ratio)
        flops, input_size, f_cin = _calc_conv2d_flops(input_size, f_cin, f_expand, kernel_size=1, stride=1, padding=0, bias=False, verbose=verbose)
        total_flops += flops
        cur_layer_id += 1

    flops, input_size, f_cin = _calc_conv2d_flops(input_size, f_cin, f_cin, kernel_size=3, stride=stride, padding=1, groups=f_cin, bias=False, verbose=verbose)
    total_flops += flops
    cur_layer_id += 1

    f_cout = make_divisible(cout * width_mults[cur_layer_id])
    flops, input_size, f_cin = _calc_conv2d_flops(input_size, f_cin, f_cout, kernel_size=1, stride=1, padding=0, bias=False, verbose=verbose)
    total_flops += flops

    return total_flops, input_size, f_cout
    





def calc_subnet_flops(model_cfg, verbose=False):
    if model_cfg['model_name'] == 'mobilenet_v1':
        input_size = model_cfg['resolution']
        width_mults = adapt_channels(model_cfg)
        cur_layer_id = -1
        total_flops = 0

        # head
        cur_layer_id += 1
        cout = make_divisible(32 * width_mults[cur_layer_id])
        flops, input_size, cin = _calc_conv2d_flops(input_size, 3, cout, kernel_size=3, stride=2, padding=1, bias=False, verbose=verbose)
        total_flops += flops

        # blocks
        cur_layer_id += 1; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=32, cout=64, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=64, cout=128, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=128, cout=128, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=128, cout=256, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=256, cout=256, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=256, cout=512, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=512, cout=1024, stride=2, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 2; flops, input_size, cin = _calc_mbv1_block_flops(input_size, cin=1024, cout=1024, stride=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        
        # pool
        flops, input_size, cin = _calc_avgpool_flops(input_size, cin, output_size=1, verbose=verbose)
        total_flops += flops
        
        # classifier
        cin = input_size**2 * cin
        flops = _calc_fc_flops(cin, 1000, verbose=verbose)
        total_flops += flops

        if verbose:
            print(f'Total flops: {total_flops/1e6:.2f}M')

        return total_flops / 1e6

    elif model_cfg['model_name'] == 'mobilenet_v2':
        
        input_size = model_cfg['resolution']
        width_mults = adapt_channels(model_cfg)
        cur_layer_id = -1
        total_flops = 0

        # head
        cur_layer_id += 1
        cout = make_divisible(32 * width_mults[cur_layer_id])
        flops, input_size, cin = _calc_conv2d_flops(input_size, 3, cout, kernel_size=3, stride=2, padding=1, bias=False, verbose=verbose)
        total_flops += flops

        # blocks
        cur_layer_id += 1; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=32, cout=16, stride=1, expand_ratio=1, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 2; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=16, cout=24, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=24, cout=24, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=24, cout=32, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=32, cout=32, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=32, cout=32, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=32, cout=64, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=64, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=96, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=96, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=96, cout=160, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=160, cout=160, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops
        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=160, cout=160, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        cur_layer_id += 3; flops, input_size, cin = _calc_mbv2_block_flops(input_size, cin=160, cout=320, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, width_mults=width_mults, verbose=verbose); total_flops += flops

        # tail
        cur_layer_id += 3
        flops, input_size, cin = _calc_conv2d_flops(input_size, cin, 1280, kernel_size=1, stride=1, padding=0, bias=False, verbose=verbose)
        total_flops += flops
        
        # pool
        flops, input_size, cin = _calc_avgpool_flops(input_size, cin, output_size=1, verbose=verbose)
        total_flops += flops

        # classifier
        cin = input_size**2 * cin
        flops = _calc_fc_flops(cin, 1000, verbose=verbose)
        total_flops += flops

        total_flops /= 1e6  # M
        if verbose:
            print(f'Total flops: {total_flops:.2f}M')
        
        return total_flops

    else:
        print('NotImplementedError')
        raise NotImplementedError

def set_active_subnet(model, model_cfg):
    width_mults = adapt_channels(model_cfg)

    model.apply(lambda m: setattr(m, 'width_mults', width_mults))
    model.apply(lambda m: setattr(m, 'infer_type_target', f"{model_cfg['infer_metric_type']}_{model_cfg['infer_metric_target']}"))
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.active_resolution = model_cfg['resolution']
    else:
        model.active_resolution = model_cfg['resolution']




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



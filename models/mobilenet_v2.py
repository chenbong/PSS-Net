import math
import torch
import torch.nn as nn
import torch.nn.functional

from .slimmable_ops import USBatchNorm2d, USConv2d, USLinear

class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand_ratio, cur_layer_id, infer_type_target_list):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.residual_connection = stride == 1 and cin == cout

        layers = []
        expand = cin * expand_ratio
        # expand
        if expand_ratio != 1:
            layers += [
                USConv2d(cin, expand, 1, 1, 0, bias=False, layer_id=cur_layer_id),
                USBatchNorm2d(expand, infer_type_target_list),
                nn.ReLU6(inplace=True),
            ]
            cur_layer_id += 1

        # depthwise + project back
        layers += [
                USConv2d(expand, expand, 3, stride, 1, groups=expand, bias=False, layer_id=cur_layer_id),
                USBatchNorm2d(expand, infer_type_target_list),
                nn.ReLU6(inplace=True),
        ]
        cur_layer_id += 1

        layers += [
                USConv2d(expand, cout, 1, 1, 0, bias=False, layer_id=cur_layer_id),
                USBatchNorm2d(cout, infer_type_target_list),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, infer_type_target_list, n_classes=1000, input_size=224):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.active_resolution = input_size
        self.features = []
        cur_layer_id = -1

        # features_head
        cur_layer_id +=1
        self.features += [
            nn.Sequential(
                USConv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False, layer_id=cur_layer_id),
                USBatchNorm2d(32, infer_type_target_list),
                nn.ReLU6(inplace=True),
            )
        ]

        # features_blocks
        cur_layer_id += 1; self.features += [InvertedResidual(cin=32, cout=16, stride=1, expand_ratio=1, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]

        cur_layer_id += 2; self.features += [InvertedResidual(cin=16, cout=24, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]
        cur_layer_id += 3; self.features += [InvertedResidual(cin=24, cout=24, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]

        cur_layer_id += 3; self.features += [InvertedResidual(cin=24, cout=32, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]
        cur_layer_id += 3; self.features += [InvertedResidual(cin=32, cout=32, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]
        cur_layer_id += 3; self.features += [InvertedResidual(cin=32, cout=32, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]

        cur_layer_id += 3; self.features += [InvertedResidual(cin=32, cout=64, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]
        cur_layer_id += 3; self.features += [InvertedResidual(cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]
        cur_layer_id += 3; self.features += [InvertedResidual(cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]
        cur_layer_id += 3; self.features += [InvertedResidual(cin=64, cout=64, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]

        cur_layer_id += 3; self.features += [InvertedResidual(cin=64, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]
        cur_layer_id += 3; self.features += [InvertedResidual(cin=96, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]
        cur_layer_id += 3; self.features += [InvertedResidual(cin=96, cout=96, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]

        cur_layer_id += 3; self.features += [InvertedResidual(cin=96, cout=160, stride=2, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]
        cur_layer_id += 3; self.features += [InvertedResidual(cin=160, cout=160, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]
        cur_layer_id += 3; self.features += [InvertedResidual(cin=160, cout=160, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]

        cur_layer_id += 3; self.features += [InvertedResidual(cin=160, cout=320, stride=1, expand_ratio=6, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list), ]

        # features_tail
        cur_layer_id += 3
        self.features += [
            nn.Sequential(
                USConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
                USBatchNorm2d(1280, infer_type_target_list),
                nn.ReLU6(inplace=True),
            )
        ]
        self.features = nn.Sequential(*self.features)

        # pool
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        # classifier
        self.classifier = nn.Sequential(USLinear(1280, n_classes))
        
        self.reset_parameters()

    def forward(self, x):
        if x.size(-1) != self.active_resolution:
            x = torch.nn.functional.interpolate(x, size=self.active_resolution, mode='bilinear', align_corners=True)
        x = self.features(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
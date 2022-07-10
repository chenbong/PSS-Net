import math
import torch
import torch.nn as nn
import torch.nn.functional

from .slimmable_ops import USBatchNorm2d, USConv2d, USLinear

def conv3x3_dw_pw(cin, cout, stride, cur_layer_id, infer_type_target_list):
    layers = []

    #* depth wise
    layers += [
        USConv2d(in_channels=cin, out_channels=cin, kernel_size=3, stride=stride, padding=1, groups=cin, bias=False, layer_id=cur_layer_id),
        USBatchNorm2d(cin, infer_type_target_list),
        nn.ReLU6(inplace=True),
    ]
    cur_layer_id += 1

    #* point wise
    layers += [
        USConv2d(in_channels=cin, out_channels=cout, kernel_size=1, stride=1, padding=0, bias=False, layer_id=cur_layer_id),
        USBatchNorm2d(cout, infer_type_target_list),
        nn.ReLU6(inplace=True),
    ]
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self, infer_type_target_list, num_class=1000, input_size=224):
        super(Model, self).__init__()
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]
        
        self.active_resolution = input_size
        self.features = []
        cur_layer_id = -1

        # head
        cur_layer_id += 1
        self.features += [
            nn.Sequential(
                USConv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False, layer_id=cur_layer_id),
                USBatchNorm2d(32, infer_type_target_list),
                nn.ReLU6(inplace=True),
            )
        ]

        # blocks
        cur_layer_id += 1; self.features += [conv3x3_dw_pw(cin=32, cout=64, stride=1, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]

        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=64, cout=128, stride=2, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]
        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=128, cout=128, stride=1, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]

        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=128, cout=256, stride=2, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]
        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=256, cout=256, stride=1, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]

        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=256, cout=512, stride=2, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]
        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]
        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]
        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]
        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]
        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=512, cout=512, stride=1, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]

        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=512, cout=1024, stride=2, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]
        cur_layer_id += 2; self.features += [conv3x3_dw_pw(cin=1024, cout=1024, stride=1, cur_layer_id=cur_layer_id, infer_type_target_list=infer_type_target_list),]
        self.features = nn.Sequential(*self.features)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        # classifier
        self.classifier = nn.Sequential(USLinear(1024, num_class))

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


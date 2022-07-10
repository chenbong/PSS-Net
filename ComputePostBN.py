import torch
import torch.nn.functional as F
from models.slimmable_ops import *
def adjust_bn_layers(module):
      
    if isinstance(module, nn.BatchNorm2d):
        module.reset_running_stats()
        module._old_momentum = module.momentum

        module.momentum = 0.1
        module._old_training = module.training
        module._old_track_running_stats = module.track_running_stats
        
        module.training = True
        module.track_running_stats = True
            
def restore_original_settings_of_bn_layers(module):
      
    if isinstance(module, nn.BatchNorm2d):
        module.momentum = module._old_momentum
        module.training = module._old_training
        module.track_running_stats = module._old_track_running_stats

def adjust_momentum(module, t):
    if isinstance(module, nn.BatchNorm2d):
        module.momentum = 1 / (t+1)

def ComputeBN(model, train_loader, num_batch=8):
    model.train()
    model.apply(adjust_bn_layers)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            model.apply(lambda m: adjust_momentum(m, batch_idx))
            _ = model(inputs)
            if not batch_idx < num_batch:
                break
    model.apply(restore_original_settings_of_bn_layers)
    model.eval()
    return model
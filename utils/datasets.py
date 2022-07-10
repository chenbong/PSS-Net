import os
import torch
import torch.utils.data
import numpy as np
from utils.config import args
from torchvision import datasets, transforms
from utils.transforms import Lighting


def fast_collate(batch, memory_format=torch.contiguous_format):
    inputs = [img[0] for img in batch]
    labels = torch.tensor([labels[1] for labels in batch], dtype=torch.int64)
    w = inputs[0].size[0]
    h = inputs[0].size[1]
    tensor = torch.zeros((len(inputs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(inputs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())

    return tensor, labels



class PrefetchedWrapper(object):
    def prefetched_loader(loader):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield inputs, labels
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            inputs = next_input
            labels = next_target

        yield inputs, labels

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        if (self.dataloader.sampler is not None and isinstance(self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler)):
            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

    def __len__(self):
        return len(self.dataloader)



def get_imagenet_pytorch_train_loader():
    crop_scale = 0.25
    jitter_param = 0.4
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(max(args.resolution_range), scale=(crop_scale, 1.0)),
        transforms.ColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
    ])
    train_set = datasets.ImageFolder(os.path.join(args.dataset_dir, 'train'), transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.data_loader_workers,
        drop_last=getattr(args, 'drop_last', False),
        collate_fn=fast_collate)

    return PrefetchedWrapper(train_loader), len(train_loader)


def get_imagenet_pytorch_val_loader():
    val_size = 256
    val_transforms = transforms.Compose([
        transforms.Resize(val_size),
        transforms.CenterCrop(max(args.resolution_range)),
    ])

    val_set = datasets.ImageFolder(os.path.join(args.dataset_dir, 'val'), transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=args.data_loader_workers,
        drop_last=getattr(args, 'drop_last', False),
        collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader), len(val_loader)

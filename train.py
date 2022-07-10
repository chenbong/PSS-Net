import importlib
import os
import time
import random
import operator
import torch
import torch.cuda
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax
import torch.multiprocessing as mp
from torch.multiprocessing import Pool

import ComputePostBN
from utils.setlogger import get_logger
from utils.model_profiling import model_profiling
from utils.config import args
from utils.datasets import get_imagenet_pytorch_train_loader, get_imagenet_pytorch_val_loader
import utils.comm as comm
from utils.comm import LatencyPredictor, BlockCfg, calc_subnet_flops, set_active_subnet, exp

from utils.subnet_sampler import ArchSampler, SubnetGenerator



saved_path = os.path.join(args.job_dir, f'{args.model_name}')
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, '{}.log'.format('test' if args.test_only else 'train')))


def init():
    calc_infer_metrics = []
    metric_target_offsets = []
    subnet_generators = []
    for i, infer_metric_type in enumerate(args.infer_metric_types):
        if infer_metric_type == 'flops':
            calc_infer_metric = calc_subnet_flops
            calc_infer_metrics.append(calc_infer_metric)
        elif 'latency'in infer_metric_type:
            latency_predictor = LatencyPredictor(lut_dir=args.lut_dirs[i])
            calc_infer_metric = latency_predictor.predict_subnet_latency
            calc_infer_metrics.append(calc_infer_metric)
        else:
            raise NotImplementedError


        metric_target_offset = args.infer_metric_target_range_starts[i] % args.infer_metric_target_steps[i]
        metric_target_offsets.append(metric_target_offset)

        subnet_generator = SubnetGenerator(args.model_name, args.resolution_range, args.resolution_step, args.width_mult_range, args.infer_metric_target_steps[i], metric_target_offset, calc_infer_metric, infer_metric_type)
        subnet_generators.append(subnet_generator)
    return calc_infer_metrics, metric_target_offsets, subnet_generators

calc_infer_metrics, metric_target_offsets, subnet_generators = init()


def gen_map_worker(worker_id, subnet_generator, worker_num, gen_map_num, infer_metric_target_start, infer_metric_target_stop, metric_target_step, workers_ret, base_seed):
    random.seed(base_seed + worker_id)
    infer_target_list = list(range(infer_metric_target_start+metric_target_step, infer_metric_target_stop-metric_target_step+1, metric_target_step))
    model_cfgs = []

    for i in range(gen_map_num // worker_num):
        model_cfg = subnet_generator.sample_subnet_within_list(infer_target_list)
        model_cfgs.append(model_cfg)
    workers_ret[worker_id] = model_cfgs


def set_random_seed():
    if hasattr(args, 'random_seed'):
        seed = args.random_seed
    else:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_model():
    infer_type_target_list = []
    for i, infer_metric_type in enumerate(args.infer_metric_types):
        infer_metric_target_start = args.infer_metric_target_range_starts[i]
        infer_metric_target_stop  = args.infer_metric_target_range_stops[i]
        metric_target_step = args.infer_metric_target_steps[i]

        tmp = list(range(infer_metric_target_start, infer_metric_target_stop+1, metric_target_step))
        infer_type_target_list += [f'{infer_metric_type}_{infer_target}' for infer_target in tmp]

    model_lib = importlib.import_module('models.'+args.model_name)
    model = model_lib.Model(infer_type_target_list, args.n_classes, input_size=max(args.resolution_range))
    return model


def get_optimizer(model):
    model_params = []
    for params in model.parameters():
        ps = list(params.size())
        if len(ps) == 4 and ps[1] != 1:
            weight_decay = args.weight_decay
        elif len(ps) == 2:  # fc
            weight_decay = args.weight_decay
        else:
            weight_decay = 0
        item = {
            'weight_decay': weight_decay,
            'momentum': args.momentum,
            'params': params, 
            'nesterov': True,
            'lr': args.lr, 
        }
        model_params.append(item)
    optimizer = torch.optim.SGD(model_params)
    return optimizer


def train(epoch, train_loader, len_train_loader, model, criterion, optimizer, lr_scheduler, arch_pools, arch_samplers):
    t_start = time.time()
    model.train()


    supernet_sample_rate = exp(max(args.supernet_p_range), min(args.supernet_p_range), epoch, args.epochs)
    t = exp(max(args.pool_softmax_t_range), min(args.pool_softmax_t_range), epoch, args.epochs)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # do max_net
        max_net = subnet_generators[0].sample_subnet(max_net=True)
        set_active_subnet(model, max_net)
        max_output = model(inputs)
        loss = criterion(max_output, labels)
        loss.backward()
        max_output_detach = max_output.detach()

        # do other widths and resolution
        infer_metric_type = random.choices(args.infer_metric_types, weights=[len(arch_pools['pools'][infer_metric_type]) for infer_metric_type in arch_pools['pools']])[0]
        idx = args.infer_metric_types.index(infer_metric_type)
        training_infer_metric_target_list = random.choices(range(arch_samplers[idx].min_infer_metric_target, arch_samplers[idx].max_infer_metric_target+1, args.infer_metric_target_steps[idx]), k=1)


        training_model_cfgs_list = []
        random_flag = random.random()
        for infer_metric_target in training_infer_metric_target_list:
            if random_flag < supernet_sample_rate or len(arch_pools['pools'][infer_metric_type][infer_metric_target]) < args.pool_size:
                args.sampler_num_sample = 1
                candidate_model_cfgs = arch_samplers[idx].sample_model_cfgs_according_to_prob(
                    infer_metric_target, n_samples=args.sampler_num_sample
                )
                my_pred_accs = []
                for model_cfg in candidate_model_cfgs:
                    set_active_subnet(model, model_cfg)

                    with torch.no_grad():
                        performance_metric_tensor = -1.0 * criterion(model(inputs), labels)
                        my_pred_accs.append(performance_metric_tensor)

                idx = 0
                performance_metric_tensor = my_pred_accs[idx]
                candidate_model_cfg = candidate_model_cfgs[idx]
                candidate_model_cfg_str = str(candidate_model_cfg)

            else:
                _prob = list(arch_pools['pools'][infer_metric_type][infer_metric_target].values())
                _prob = np.array(_prob)

                prob = softmax(_prob/t)
                [candidate_model_cfg_str] = random.choices(list(arch_pools['pools'][infer_metric_type][infer_metric_target].keys()), weights=prob)

                candidate_model_cfg = eval(candidate_model_cfg_str)
                set_active_subnet(model, candidate_model_cfg)

                with torch.no_grad():
                    performance_metric_tensor = -1.0 * criterion(model(inputs), labels)

            assert infer_metric_target == candidate_model_cfg['infer_metric_target']

            training_model_cfgs_list.append(candidate_model_cfg)
            if candidate_model_cfg_str not in arch_pools['pools'][infer_metric_type][infer_metric_target]:
                arch_pools['pools'][infer_metric_type][infer_metric_target][candidate_model_cfg_str] = performance_metric_tensor.item()  # performance_metric: higher is bertter
            else:
                arch_pools['pools'][infer_metric_type][infer_metric_target][candidate_model_cfg_str] = arch_pools['pools'][infer_metric_type][infer_metric_target][candidate_model_cfg_str]*args.metric_lambda + performance_metric_tensor.item()*(1-args.metric_lambda)

            if len(arch_pools['pools'][infer_metric_type][infer_metric_target]) > arch_pools['max_size']:
                min_value_key = min(arch_pools['pools'][infer_metric_type][infer_metric_target].keys(), key=(lambda k: arch_pools['pools'][infer_metric_type][infer_metric_target][k]))
                logger.info(f"infer_metric_target: {infer_metric_target}, push: {performance_metric_tensor.item()}, pop: {arch_pools['pools'][infer_metric_type][infer_metric_target][min_value_key]}")
                arch_pools['pools'][infer_metric_type][infer_metric_target].pop(min_value_key)

        for arch_id in range(2):
            if arch_id == 1:
            # do min_net
                min_net = subnet_generators[0].sample_subnet(min_net=True)
                set_active_subnet(model, min_net)
            else:
                # do middle_net
                set_active_subnet(model, training_model_cfgs_list[arch_id])
            output = model(inputs)

            loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), F.softmax(max_output_detach, dim=1))


            loss.backward()

        optimizer.step()
        lr_scheduler.step()

        if not os.path.exists(args.job_dir):
            logger.info(f'{args.job_dir} not exist...')
            raise NotImplementedError

        if batch_idx % args.print_freq == 0 or batch_idx == len_train_loader-1:
            with torch.no_grad():
                indices = torch.max(max_output, dim=1)[1]
                acc = (indices == labels).sum().cpu().numpy() / indices.size()[0]
                logger.info('TRAIN {:.1f}s LR:{:.4f} {}x Epoch:{}/{} Iter:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                    time.time() - t_start, optimizer.param_groups[0]['lr'], str(max(args.width_mult_range)), epoch,
                    args.epochs, batch_idx, len_train_loader, loss, acc)
                )
                t_start = time.time()


def validate(epoch, val_loader, model, criterion, train_loader, subnets_to_be_evaluated):
    t_start = time.time()
    model.eval()
    wandb_log_dict = {}
    with torch.no_grad():
        for infer_metric_type in subnets_to_be_evaluated:
            for key in subnets_to_be_evaluated[infer_metric_type]:
                model_cfg = subnets_to_be_evaluated[infer_metric_type][key]
                set_active_subnet(model, model_cfg)
                model = ComputePostBN.ComputeBN(model, train_loader)

                loss, acc, cnt = 0, 0, 0
                for _, (inputs, labels) in enumerate(val_loader):
                    output = model(inputs)
                    loss += criterion(output, labels).cpu().numpy() * labels.size()[0]
                    indices = torch.max(output, dim=1)[1]
                    acc += (indices == labels).sum().cpu().numpy()
                    cnt += labels.size()[0]
                logger.info(f'VAL:{model_cfg}')
                logger.info(f"VAL:{time.time() - t_start:.1f}s, id:{infer_metric_type}_{key}, infer_metric:{model_cfg['infer_metric']:.2f}, Epoch:{epoch}/{args.epochs}, Loss:{loss/cnt:.4f}, Acc:{acc/cnt:.4f}")
                t_start = time.time()
                arch_key = f"{infer_metric_type}_{model_cfg['infer_metric_target']}_acc1"
                if arch_key in wandb_log_dict:
                    wandb_log_dict[arch_key] = max(acc/cnt, wandb_log_dict[arch_key])
                else:
                    wandb_log_dict[arch_key] = acc/cnt





def train_val(arch_pools, arch_samplers):
    """train and val"""
    set_random_seed()
    model = get_model()
    model_wrapper = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    train_loader, len_train_loader  = get_imagenet_pytorch_train_loader()
    val_loader, len_val_loader = get_imagenet_pytorch_val_loader()

    optimizer = get_optimizer(model_wrapper)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len_train_loader*args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler.last_epoch = last_epoch * len_train_loader

        arch_pools = checkpoint['arch_pools']
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_random_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])

        logger.info(f"Loaded checkpoint '{args.resume}' at epoch '{last_epoch}'")
    else:
        last_epoch = -1
        # profiling
        for subnet_generator, calc_infer_metric in zip(subnet_generators, calc_infer_metrics):
            max_net = subnet_generator.sample_subnet(max_net=True)
            model_profiling(model, max_net, use_cuda=True, print_=True, verbose=False)
            logger.info(f'{calc_infer_metric(max_net, verbose=False)}')

            min_net = subnet_generator.sample_subnet(min_net=True)
            model_profiling(model, min_net, use_cuda=True, print_=True, verbose=False)
            logger.info(f'{calc_infer_metric(min_net, verbose=False)}')

    if args.test_only:
        logger.info('Start testing.')
        subnets_to_be_evaluated = {}
        for idx, infer_metric_type in enumerate(args.infer_metric_types):
            subnets_to_be_evaluated[infer_metric_type] = {}
            subnets_to_be_evaluated[infer_metric_type]["max_net"] = subnet_generators[idx].sample_subnet(max_net=True)
            for infer_metric_target in arch_pools['infer_metric_target_list'][infer_metric_type]:
                logger.info(f"=== {infer_metric_type}_{infer_metric_target}, {len(arch_pools['pools'][infer_metric_type][infer_metric_target])}")
                pool_list = list(arch_pools['pools'][infer_metric_type][infer_metric_target].items())
                pool_list.sort(key=operator.itemgetter(1), reverse=True)
                logger.info(f"=> {pool_list[:args.valid_topk]}")
                subnets_to_be_evaluated[infer_metric_type][f'{infer_metric_target}'] = eval(pool_list[0][0])
                for i in range(1, args.valid_topk):
                    subnets_to_be_evaluated[infer_metric_type][f'{infer_metric_target}[{i}]'] = eval(pool_list[i][0])

            subnets_to_be_evaluated[infer_metric_type]['min_net'] = subnet_generators[idx].sample_subnet(min_net=True)
        logger.info(len(subnets_to_be_evaluated[infer_metric_type]))

        validate(last_epoch, val_loader, model_wrapper, criterion, train_loader, subnets_to_be_evaluated)
        return

    logger.info('Start training.')
    for epoch in range(last_epoch + 1, args.epochs):
        train(epoch, train_loader, len_train_loader, model_wrapper, criterion, optimizer, lr_scheduler, arch_pools, arch_samplers)
        subnets_to_be_evaluated = {}
        for idx, infer_metric_type in enumerate(args.infer_metric_types):
            subnets_to_be_evaluated[infer_metric_type] = {}
            subnets_to_be_evaluated[infer_metric_type]["max_net"] = subnet_generators[idx].sample_subnet(max_net=True)
            for infer_metric_target in arch_pools['infer_metric_target_list'][infer_metric_type]:
                logger.info(f"=== {infer_metric_type}_{infer_metric_target}, {len(arch_pools['pools'][infer_metric_type][infer_metric_target])}")
                pool_list = list(arch_pools['pools'][infer_metric_type][infer_metric_target].items())
                pool_list.sort(key=operator.itemgetter(1), reverse=True)
                logger.info(f"=> {pool_list[:args.valid_topk]}")
                if epoch+1 == args.epochs:
                    subnets_to_be_evaluated[infer_metric_type][f'{infer_metric_target}'] = eval(pool_list[0][0])
                    for i in range(1, args.valid_topk):
                        subnets_to_be_evaluated[infer_metric_type][f'{infer_metric_target}[{i}]'] = eval(pool_list[i][0])

            if epoch+1 == args.epochs:
                subnets_to_be_evaluated[infer_metric_type]['min_net'] = subnet_generators[idx].sample_subnet(min_net=True)

        validate(epoch, val_loader, model_wrapper, criterion, train_loader, subnets_to_be_evaluated)
        torch.save(
            {
                'model': model_wrapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'last_epoch': epoch,
                'arch_pools': arch_pools,
                'random_state' : random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_rng_state' : torch.get_rng_state(),
                'torch_cuda_rng_state': torch.cuda.get_rng_state(),
            },
            os.path.join(saved_path, 'checkpoint_{:03d}.pt'.format(epoch)))

    return


def main():
    if args.gpu_ids:
        gpu_ids_str = ','.join(str(i) for i in args.gpu_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
        logger.info(f'CUDA_VISIBLE_DEVICES={gpu_ids_str}')


    if args.test_only:
        arch_pools = None
        arch_samplers = None
    else:
        arch_samplers = []
        for i, infer_metric_type in enumerate(args.infer_metric_types):
            map_path = os.path.join(args.job_dir, f'{infer_metric_type}.map')
            worker_num = 64
            p = Pool(worker_num)
            manager = mp.Manager()
            workers_ret = manager.dict()
            start = time.time()
            for j in range(worker_num):
                p.apply_async(gen_map_worker, args=(j, subnet_generators[i], worker_num, args.gen_map_num, args.infer_metric_target_range_starts[i], args.infer_metric_target_range_stops[i], args.infer_metric_target_steps[i], workers_ret, args.random_seed))

            logger.info('Waiting for all sub worker done...')
            p.close()
            p.join()
            logger.info(f'Gen infer_metric map done, cost: {time.time()-start:.2f}s')
            
            start = time.time()
            with open(map_path, 'w') as f:
                for worker_i in range(len(workers_ret)):
                    for model_cfg in workers_ret[worker_i]:
                        f.write(f'{model_cfg}\n')
            logger.info(f'Write map file done, cost: {time.time()-start:.2f}s')

            # build model_cfg sampler
            start = time.time()
            arch_sampler = ArchSampler(args.model_name, map_path, args.infer_metric_target_steps[i], metric_target_offsets[i], calc_infer_metrics[i], args.infer_metric_types[i])
            logger.info(f'=> min infer_metric target: {arch_sampler.min_infer_metric_target}, max_infer_metric: {arch_sampler.max_infer_metric_target}, build arch_sampler cost:{time.time()-start:.2f}s')

            arch_samplers.append(arch_sampler)


        arch_pools = {}
        arch_pools['max_size'] = args.pool_size
        arch_pools['infer_metric_target_list'] = {}
        for i, arch_sampler in enumerate(arch_samplers):
            targets = sorted(arch_sampler.prob_map['infer_metric'].keys(), reverse=True)
            arch_pools['infer_metric_target_list'][args.infer_metric_types[i]] = targets
        arch_pools['pools'] = {}
        for infer_metric_type in args.infer_metric_types:
            arch_pools['pools'][infer_metric_type] = {}
            for target in arch_pools['infer_metric_target_list'][infer_metric_type]:
                arch_pools['pools'][infer_metric_type][target] = {}

    train_val(arch_pools, arch_samplers)


if __name__ == "__main__":
    main()

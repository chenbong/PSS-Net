import argparse

parser = argparse.ArgumentParser(description='PSS-Net')
parser.add_argument('--dataset_dir', type=str, default=None)
parser.add_argument('--data_loader_workers', type=int, default=8)
parser.add_argument('--n_classes', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--model_name', type=str, default='mobilenet_v2', choices=['mobilenet_v1', 'mobilenet_v2'])
parser.add_argument('--job_dir', default=None, type=str)
parser.add_argument('--gpu_ids', default=[2,3], type=int, nargs='+')
parser.add_argument('--lr', default=0.3, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--resume', type=str, default=None)

parser.add_argument('--infer_metric_types', type=str, default=['flops'], nargs='+', choices=['flops', 'latency_gpu', 'latency_cpu'])
parser.add_argument('--infer_metric_target_range_starts', type=int, default=[60], nargs='+')
parser.add_argument('--infer_metric_target_range_stops',  type=int, default=[300], nargs='+')
parser.add_argument('--infer_metric_target_steps', type=int, default=[10], nargs='+')
parser.add_argument('--lut_dirs', type=str, default=[None], nargs='+')

parser.add_argument('--gen_map_num', type=int, default=100000)
parser.add_argument('--pool_size', default=50, type=int)
parser.add_argument('--supernet_p_range', default=[1.0, 1e-2], nargs='+', type=float)
parser.add_argument('--pool_softmax_t_range', default=[1.0, 1e-2], nargs='+', type=float)
parser.add_argument('--metric_lambda', default=0.9, type=float)
parser.add_argument('--valid_topk', default=5, type=int)
parser.add_argument('--resolution_range', type=int, default=[128, 224], nargs='+')
parser.add_argument('--resolution_step', type=int, default=8)
parser.add_argument('--width_mult_range', type=float, default=[0.75, 1.0], nargs='+')
args = parser.parse_args()




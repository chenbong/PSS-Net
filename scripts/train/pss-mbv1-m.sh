python train.py \
--dataset_dir [DATASET_DIR] \
--job_dir [JOB_DIR] \
--model_name mobilenet_v1 \
--gpu_ids 0 1 2 3 \
--batch_size 1024 \
--epochs 250 \
--infer_metric_types flops latency_gpu latency_cpu \
--infer_metric_target_range_starts 110 70 10000 \
--infer_metric_target_range_stops 570 240 31000 \
--infer_metric_target_steps 10 10 1000 \
--lut_dirs None lut/mbv1_gpu.pkl lut/mbv1_cpu.pkl \
--gen_map_num 1000000


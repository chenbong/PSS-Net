python train.py \
--dataset_dir [DATASET_DIR] \
--job_dir [JOB_DIR] \
--model_name mobilenet_v1 \
--gpu_ids 0 1 2 3 \
--batch_size 1024 \
--epochs 250 \
--infer_metric_types latency_cpu \
--infer_metric_target_range_starts 10000 \
--infer_metric_target_range_stops 31000 \
--infer_metric_target_steps 1000 \
--lut_dirs lut/mbv1_cpu.pkl


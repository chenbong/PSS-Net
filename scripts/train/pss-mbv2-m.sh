python train.py \
--dataset_dir [DATASET_DIR] \
--job_dir [JOB_DIR] \
--model_name mobilenet_v2 \
--gpu_ids 0 1 2 3 \
--batch_size 1024 \
--epochs 250 \
--infer_metric_types flops latency_gpu latency_cpu \
--infer_metric_target_range_starts 70 100 12000 \
--infer_metric_target_range_stops 300 270 28000 \
--infer_metric_target_steps 10 10 1000 \
--lut_dirs None lut/mbv2_gpu.pkl lut/mbv2_cpu.pkl



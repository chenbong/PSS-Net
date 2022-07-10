python train.py \
--dataset_dir [DATASET_DIR] \
--job_dir [JOB_DIR] \
--model_name mobilenet_v2 \
--gpu_ids 0 1 2 3 \
--batch_size 1024 \
--epochs 250 \
--infer_metric_types latency_cpu \
--infer_metric_target_range_starts 12000 \
--infer_metric_target_range_stops 28000 \
--infer_metric_target_steps 1000 \
--gen_map_num 100000 \
--lut_dirs lut/mbv2_cpu.pkl \
--resume [RESUME] \
--test_only


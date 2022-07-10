python train.py \
--dataset_dir [DATASET_DIR] \
--job_dir [JOB_DIR] \
--model_name mobilenet_v2 \
--gpu_ids 0 1 2 3 \
--batch_size 1024 \
--epochs 250 \
--infer_metric_types flops \
--infer_metric_target_range_starts 70 \
--infer_metric_target_range_stops 300 \
--infer_metric_target_steps 10



#!/bin/bash
model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
output_act_scales_file_path="act_scales/llama_3.2_3b.pt"
num_samples=512
sequence_length=512
path_to_the_calibration_dataset="datasets/val.jsonl.zst"

python dependency/examples/generate_act_scales.py \
    --model-name $model_name_or_path \
    --output-path $output_act_scales_file_path \
    --num-samples $num_samples \
    --seq-len $sequence_length \
    --dataset-path $path_to_the_calibration_dataset
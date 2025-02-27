#!/bin/bash

module load anaconda3/2024.02-1
conda activate model_acc

python 1_models/llama3_model.py
# cd 3_quant/smooth_quant_demo
# sh gen_act_scales.sh
# python smooth_quant_demo.py


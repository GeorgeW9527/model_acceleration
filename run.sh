#!/bin/bash

module load anaconda3/2024.02-1
conda activate model_acc

python models/llama3_model.py

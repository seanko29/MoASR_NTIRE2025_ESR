#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_demo.py \
    --data_dir /home/sean/ntire_data/real_validation \
    --save_dir /home/sean/SR_project/ACCV/NTIRE2025_ESR/results \
    --model_id 11       
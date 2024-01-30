#!/bin/bash

# Path of the pre-trained model.
MODEL_PATH=../pre_trained_models/squeeze_complex_bypass.pt

# Where the dataset is saved to.
DATASET_DIR=../../data


# Run Evaluation.
python ../eval_model.py \
  --model_path=${MODEL_PATH} \
  --dataset_path=${DATASET_DIR} \
  --model_name=Squeeze_complex_bypass \
  --batch_size=100 \
  --trec=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
  --L=9,96,8,48,64,8,72,64,16,144,128,4,144,32,6,54,48,6,216,8,4,288,256,8,288,4 \
  --H=5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 \
  --bp_trec=1,0,0,0 \
  --bp_L=12,4,32,128 \
  --bp_H=5,8,5,5 \
  --gpu=0

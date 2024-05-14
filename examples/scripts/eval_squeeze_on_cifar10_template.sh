#!/bin/bash

MODEL_PATH=pre_trained_models/squeeze.pt
DATASET_DIR=data

python eval_model.py \
--model_path=${MODEL_PATH} \
--dataset_path=${DATASET_DIR} \
--model_name=SqueezeNet \
--batch_size=100 \
--trec=0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
--L=9,96,8,48,64,8,72,64,16,144,128,4,144,32,6,54,48,6,216,8,4,288,256,8,288,4 \
--H=5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 \
--gpu=0

echo "===================="

python eval_model.py \
--model_path=${MODEL_PATH} \
--dataset_path=${DATASET_DIR} \
--model_name=SqueezeNet \
--batch_size=100 \
--trec=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
--gpu=0
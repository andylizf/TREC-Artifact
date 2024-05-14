#!/bin/bash

MODEL_PATH=pre_trained_models/cifar.pt
DATASET_DIR=data

python eval_model.py \
--model_path=${MODEL_PATH} \
--dataset_path=${DATASET_DIR} \
--model_name=CifarNet \
--batch_size=100 \
--trec=0,1 \
--L=5,10 \
--H=15,10 \
--gpu=0

echo "===================="

python eval_model.py \
--model_path=${MODEL_PATH} \
--dataset_path=${DATASET_DIR} \
--model_name=CifarNet \
--batch_size=100 \
--trec=0,0 \
--gpu=0
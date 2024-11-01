#!/bin/bash

CHECKPOINT_PATH=EXP
DATASET_DIR=data

python train_model.py \
--checkpoint_path=${CHECKPOINT_PATH} \
--dataset_path=${DATASET_DIR} \
--model_name=densenet \
--epochs=100 \
--batch_size=64 \
--learning_rate=0.1 \
--momentum=0.9 \
--weight_decay=1e-4 \
--depth=40 \
--k=12 \
--trec=0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
--L=9,\
3,9,3,9,3,9,3,9,3,9,3,9,\
6,\
6,9,6,9,6,9,6,9,6,9,6,9,\
6,\
6,9,6,9,6,9,6,9,6,9,6,9 \
--H=8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 \
--gpu=0

echo "==============================="

# 运行没有TREC的基准模型
python train_model.py \
--checkpoint_path=${CHECKPOINT_PATH}_baseline \
--dataset_path=${DATASET_DIR} \
--model_name=densenet \
--epochs=100 \
--batch_size=64 \
--learning_rate=0.1 \
--momentum=0.9 \
--weight_decay=1e-4 \
--depth=40 \
--k=12 \
--trec=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
--gpu=0
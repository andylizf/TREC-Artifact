#!/bin/bash

TRAIN_DIR=EXP
DATASET_DIR=data

python train_model.py \
--checkpoint_path=${TRAIN_DIR} \
--dataset_path=${DATASET_DIR} \
--model_name=Cifarnet \
--epochs=50 \
--batch_size=10 \
--learning_rate=0.001 \
--momentum=0.95 \
--weight_decay=0.0001 \
--grad_clip=5 \
--step=15 \
--trec=0,1 \
--L=5,10 \
--H=15,10 \
--gpu=0

echo "===================="

# python train_model.py \
# --checkpoint_path=${TRAIN_DIR} \
# --dataset_path=${DATASET_DIR} \
# --model_name=Cifarnet \
# --epochs=50 \
# --batch_size=10 \
# --learning_rate=0.001 \
# --momentum=0.95 \
# --weight_decay=0.0001 \
# --grad_clip=5 \
# --step=15 \
# --trec=0,0 \
# --L=5,10 \
# --H=15,10 \
# --gpu=0
#!/bin/bash

TRAIN_DIR=EXP
DATASET_DIR=data


# Run training.
python ../train_model.py \
--checkpoint_path=${TRAIN_DIR} \
--dataset_path=${DATASET_DIR} \
--model_name=SqueezeNet \
--epochs=50 \
--batch_size=10 \
--learning_rate=0.001 \
--momentum=0.95 \
--weight_decay=0.0001 \
--grad_clip=5 \
--step=15 \
--trec=0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
--L=9,96,8,48,64,8,72,64,16,144,128,4,144,32,6,54,48,6,216,8,4,288,256,8,288,4 \
--H=5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 \
--gpu=0

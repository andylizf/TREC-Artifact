#!/bin/bash

CHECKPOINT_PATH=autoencoder_exp
DATASET_DIR=data

# Train autoencoder with TREC
python train_autoencoder.py \
--model_name=autoencoder_trec \
--checkpoint_path=${CHECKPOINT_PATH} \
--dataset_path=${DATASET_DIR} \
--epochs=100 \
--batch_size=32 \
--learning_rate=1e-4 \
--weight_decay=1e-6 \
--trec=0,0,0,0,0,0,0,0,\
0,0,0,1,0,0,0,0 \
--L=9,9,9,9,9,9,9,9,\
9,9,9,9,9,9,9,9 \
--H=8,8,8,8,8,8,8,8,\
8,8,8,8,8,8,8,8 \
--gpu=0

echo "==============================="

# Train baseline autoencoder without TREC
python train_autoencoder.py \
--model_name=autoencoder_trec \
--checkpoint_path=${CHECKPOINT_PATH}_baseline \
--dataset_path=${DATASET_DIR} \
--epochs=100 \
--batch_size=32 \
--learning_rate=1e-4 \
--weight_decay=1e-6 \
--gpu=0
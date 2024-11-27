#!/bin/bash

CHECKPOINT_PATH=autoencoder_exp
DATASET_DIR=data

# Calculate TREC parameters for autoencoder
# 4 downsampling blocks in encoder (8 conv layers) and 4 upsampling blocks in decoder (8 conv layers)
# Total: 16 TREC layers

# python train_model.py \
# --checkpoint_path=${CHECKPOINT_PATH} \
# --dataset_path=${DATASET_DIR} \
# --model_name=autoencoder_trec \
# --epochs=100 \
# --batch_size=32 \
# --learning_rate=1e-4 \
# --weight_decay=1e-6 \
# --trec=1,1,1,1,1,1,1,1,\
# 1,1,1,1,1,1,1,1 \
# --L=9,9,9,9,9,9,9,9,\
# 9,9,9,9,9,9,9,9 \
# --H=8,8,8,8,8,8,8,8,\
# 8,8,8,8,8,8,8,8 \
# --gpu=0

echo "==============================="

# Run baseline model without TREC
python train_model.py \
--checkpoint_path=${CHECKPOINT_PATH}_baseline \
--dataset_path=${DATASET_DIR} \
--model_name=autoencoder_trec \
--epochs=100 \
--batch_size=32 \
--learning_rate=1e-4 \
--weight_decay=1e-6 \
--gpu=0 
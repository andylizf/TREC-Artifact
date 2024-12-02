#!/bin/bash

python examples/train_model.py \
--checkpoint_path=EXP_squeeze_L7 \
--model_name=SqueezeNet \
--epochs=50 \
--batch_size=128 \
--learning_rate=0.001 \
--momentum=0.95 \
--weight_decay=0.0001 \
--grad_clip=5 \
--step=15 \
--trec=1 \
--L=7 \
--H=8

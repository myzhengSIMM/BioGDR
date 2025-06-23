#!/bin/bash

python train.py \
--gpu 0 \
--dataset_dir 'data/train_example/' \
--save_dir 'data/train_example/results/train_example' \
--num_workers 4 \
--batch_size 256 \
--epochs 300 \
--early_stop_epoch 10 \
--init_lr 0.0001 \
--weight_decay 0.00001

# If no GPU device is available, you can specify --cpu to force the model to run on the CPU.
# specify --get_attn to get gene and pathway attention score.
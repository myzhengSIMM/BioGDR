#!/bin/bash
python ind_eval.py \
--gpu 0 \
--result_dir 'data/train_example/results/train_example' \
--target_data_dir 'data/eval_example' \
--save_dir 'data/eval_example/results/train_example_eval_example' \
--num_workers 4 

# If no GPU device is available, you can specify --cpu to force the model to run on the CPU.
# specify --get_attn to get gene and pathway attention score.
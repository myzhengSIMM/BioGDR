# BioGDR
The code of "Multimodal interpretable deep learning for transcriptome-informed precision oncology and drug mechanism analysis".

Precision oncology faces critical challenges in interpreting complex cellular signals and predicting drug responses across heterogeneous cancer environments. Here, we present BioGDR, a multimodal interpretable deep learning framework that integrates structure-based predicted biological features (differential gene expression and kinase inhibition profiles), eliminating the need for experimental measurements. By modeling tumor transcriptomic states through pathway-informed graph neural networks and employing a drug-guided attention strategy, BioGDR enables mechanistic insights into drug sensitivity across compound and cellular contexts. Comprehensive evaluations demonstrate that BioGDR outperforms existing methods in compound screening, cell line sensitivity prediction, and clinical outcome assessment. Experimental validation with a novel ALDH1B1 inhibitor confirms its ability to identify sensitive cell populations and reveal underlying mechanisms. This work establishes a robust, biologically-informed framework bridging preclinical drug development and clinical applications, advancing precision oncology through integrative multimodal learning and interpretable mechanism analysis.

![Figure1](https://github.com/user-attachments/assets/14a5a3b1-c6b0-4b3b-8f5f-03812bfa3926)

## Requirements
  - python=3.8
  - pytorch-cuda=11.7
  - pytorch=1.11.0
  - rdkit==2020.09.01
  - scikit-learn==1.0.2
  - dgl-cuda11.3==0.8.1
  - dgllife==0.2.9
environment.yml contains environment of this project.

## Model training
To train BioGDR, run either:
```
bash train.sh
```
or
```
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
```
If no GPU device is available, you can specify --cpu to force the model to run on the CPU. 
For example:
```
python train.py \
--cpu \
--dataset_dir 'data/train_example/' \
--save_dir 'data/train_example/results/train_example' \
--num_workers 4 \
--batch_size 256 \
--epochs 300 \
--early_stop_epoch 10 \
--init_lr 0.0001 \
--weight_decay 0.00001
```
Specify --get_attn to get gene and pathway attention score.

## Model prediction
Run either:
```
bash ind_eval.sh
```
or
```
python ind_eval.py \
--gpu 0 \
--result_dir 'data/train_example/results/train_example' \
--target_data_dir 'data/eval_example' \
--save_dir 'data/eval_example/results/train_example_eval_example' \
--num_workers 4
```
You can also use --cpu to specify cpu device and --get_attn to get attention score.

# BioGDR
The code of "Multimodal interpretable deep learning for transcriptome-informed precision oncology and drug mechanism analysis".

Precision oncology faces critical challenges in interpreting complex cellular signals and predicting drug responses across heterogeneous cancer environments. Here, we present BioGDR, a multimodal interpretable deep learning framework that integrates structure-based predicted biological features differential gene expression and kinase inhibition profiles, eliminating the need for experimental measurements. By modeling tumor transcriptomic states through pathway-informed graph neural networks and employing a drug-guided attention strategy, BioGDR enables mechanistic insights into drug sensitivity across compound and cellular contexts. Comprehensive evaluations demonstrate that BioGDR outperforms existing methods in compound screening relevant to early-stage drug discovery and in predicting cell-line sensitivity across heterogeneous cellular states characteristic of precision oncology, while analyses on clinical patient cohorts further confirm its practical utility and generalization capability. Experimental validation with a novel ALDH1B1 inhibitor confirms its ability to identify sensitive cell populations and reveal underlying mechanisms. This work establishes a robust, biologically-informed framework bridging preclinical drug development and clinical applications, advancing precision oncology through integrative multimodal learning and interpretable mechanism analysis.

<img width="4056" height="4713" alt="BioGDR" src="https://github.com/user-attachments/assets/6d68c9f1-6d3c-45a3-9ca3-e1b65bf22a93" />


## Requirements
  - python=3.8
  - pytorch-cuda=11.7
  - pytorch=1.11.0
  - rdkit==2020.09.01
  - scikit-learn==1.0.2
  - dgl-cuda11.3==0.8.1
  - dgllife==0.2.9

environment.yml contains environment of this project.


## Data availability
### Cell expression, pathway information and protein-protein interaction data
The RNA-Seq data can be downloaded from Depmap portal as the file named “CCLE_expression.csv” in 22Q1 version (https://depmap.org/portal/data_page/?tab=allData). 
The KEGG pathway information is available from the following link: https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#C2. 
The protein-protein interaction data can be downloaded from https://string-db.org/cgi/download. 

### Drug predicted biological features
include DE and KI features, deposited to Zenodo at https://zenodo.org/records/15718571.

### Drug sensitivity data
The drug sensitivity and compound data were also retrieved from Depmap portal for file “secondary-screen-dose-response-curve-parameters.csv” and "secondary-screen-replicate-collapsed-treatment-info.csv" in PRISM Repurposing Secondary Screen 19Q4 (https://depmap.org/portal/data_page/?tab=allData).

### AML dataset
The expression and other information in the AML dataset were source from the paper: Wang et al. Integrative proteogenomic and pharmacological landscape of acute myeloid leukaemia. Sci. Bull. 70, 1051-1056 (2025).

### TCGA dataset
The sensitivity data for the TCGA dataset were obtained from supplementary data of the study: Ding, Z., Zu, S. & Gu, J. Evaluating the molecule-based prediction of clinical drug responses in cancer. Bioinformatics 32, 2891-2895 (2016). 
The endpoint information for survival analysis were sourced from Liu, J. et al. An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. Cell 173, 400-416.e411 (2018), which are available in their Table S1.


## Data processing and partitioning
Download the following files from the provided links:
- CCLE_expression.csv
- secondary-screen-dose-response-curve-parameters.csv
- secondary-screen-replicate-collapsed-treatment-info.csv

Place the downloaded files into the ./data/RawFile. Then, run the ./data/data_processing.ipynb script to generate and save the processed dataset under ./data/ProcessedFile.
Using the processed dataset, you can execute the ./data/data_partitioning.ipynb script to generate data splits according to different partitioning strategies.


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

Due to file size limitations, the complete pathway dataset is hosted on Zenodo: [10.5281/zenodo.15718571](https://doi.org/10.5281/zenodo.15718571).







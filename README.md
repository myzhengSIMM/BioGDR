# BioGDR
The code of "Multimodal interpretable deep learning for transcriptome-informed precision oncology and drug mechanism analysis".

Precision oncology faces critical challenges in interpreting complex cellular signals and predicting drug responses across heterogeneous cancer environments. Here, we present BioGDR, a multimodal interpretable deep learning framework that integrates structure-based predicted biological features differential gene expression and kinase inhibition profiles, eliminating the need for experimental measurements. By modeling tumor transcriptomic states through pathway-informed graph neural networks and employing a drug-guided attention strategy, BioGDR enables mechanistic insights into drug sensitivity across compound and cellular contexts. Comprehensive evaluations demonstrate that BioGDR outperforms existing methods in compound screening relevant to early-stage drug discovery and in predicting cell-line sensitivity across heterogeneous cellular states characteristic of precision oncology, while analyses on clinical patient cohorts further confirm its practical utility and generalization capability. Experimental validation with a novel ALDH1B1 inhibitor confirms its ability to identify sensitive cell populations and reveal underlying mechanisms. This work establishes a robust, biologically-informed framework bridging preclinical drug development and clinical applications, advancing precision oncology through integrative multimodal learning and interpretable mechanism analysis.

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

## Data availability
### Cell expression, pathway information and protein-protein interaction data
The RNA-Seq data can be downloaded from Depmap portal as the file named “CCLE_expression.csv” in 22Q1 version (https://depmap.org/portal/data_page/?tab=allData). 
The KEGG pathway information is available from the following link: https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#C2. 
The protein-protein interaction data can be downloaded from https://string-db.org/cgi/download. 

### Drug predicted biological features
include DE and KI features, deposited to Zenodo at https://zenodo.org/records/15718571.

### Drug sensitivity data
The drug sensitivity and compound data were also retrieved from Depmap portal for file “secondary-screen-dose-response-curve-parameters.csv” in PRISM Repurposing Secondary Screen 19Q4 (https://depmap.org/portal/data_page/?tab=allData&releasename=PRISM%20Repurposing%2019Q4&filename=secondary-screen-dose-response-curve-parameters.csv).

### AML dataset
The expression and other information in the AML dataset were source from the paper: Wang et al. Integrative proteogenomic and pharmacological landscape of acute myeloid leukaemia. Sci. Bull. 70, 1051-1056 (2025).

### TCGA dataset
The sensitivity data for the TCGA dataset were obtained from supplementary data of the study: Ding, Z., Zu, S. & Gu, J. Evaluating the molecule-based prediction of clinical drug responses in cancer. Bioinformatics 32, 2891-2895 (2016). 
The endpoint information for survival analysis were sourced from Liu, J. et al. An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. Cell 173, 400-416.e411 (2018), which are available in their Table S1.


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

## Implementation of baseline models
- The DNN and is built directly on the PyTorch framework.
- All other implemented baseline models were sourced from their original code repositories.
- To ensure environment integrity and avoid dependency conflicts, a dedicated conda environment was created for each baseline model based on the environment file (such as environment.yml or requirements.txt) provided by the respective authors.
- For models with pretrained modules, we loaded the specific version reported to have the best performance in their original publications.
- We strictly adhered to each model's original hyperparameters. The only modification was to the data loading pipeline to accommodate our standardized input format and the data partitions generated by data/data_split.ipynb.
- Each model was then trained and evaluated across all testing scenarios using these standardized data splits.

**CSG2A**
Repository: https://github.com/eugenebang/CSG2A
- Used the provided pretrained weights and fine-tuned the model on GDSC or PRISM datasets.
- The 978-dimensional feature version was chosen to maintain comparable feature space dimensions with BioGDR.
- Adapted the data loader to read the standardized split files.

**CANDELA**
Repository: https://zenodo.org/doi/10.5281/zenodo.8020945
- Used single-task cross-attention version pretrained on metabolite property prediction, which was reported as the best-performing configuration in the original publication.
- Adapted the data loader to read the standardized split files.

**DTLCDR**
Repository: https://github.com/yujie0317/DTLCDR
- Used the trained GCADTI model to predict target features for each drug in response datasets.
- Cell line expression profiles were processed identically to the original preprocessing code.
- Adapted the data loader to read the standardized split files.
- Clinical Scenario: the trained DTLCDR was used directly for inference without retraining, as the zero-shot scenario design did not allow for model training on the target patient data.

**DeepTTA**
Repository: https://github.com/jianglikun/DeepTTC
•	Followed the standard data processing and alignment pipeline as described in the original repository.
•	Adapted the data loader to read the standardized split files.

**HiDRA**
Repository: https://github.com/GIST-CSBL/HiDRA
- Used HiDRA_FeatureGeneration.ipynb to process cell line expression profiles, which is consistent with the pathway information used in BioGDR.
- Drug features were limited to molecular fingerprints following the description in the original manuscript.
- Adapted the data loader to read the standardized split files.

**Precily**
Repository: https://github.com/SmritiChawla/Precily
- Followed repository instructions for feature generation and model setup.
- Used the pretrained SMILESVec model to obtain drug embeddings.
- Used GSVA algorithm to obtain pathway enrichment scores for cell lines.
- Adapted the data loader to read the standardized split files.





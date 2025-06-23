#Get path parameters
import os

def get_path_config(args):
    args.geneset_path = os.path.join(args.dataset_dir,"geneset_info/c2.cp.kegg.v2022.1.Hs.entrez.gmt")

    get_prism_drug_path(args)
    args.exp_path = os.path.join(args.dataset_dir, 'cell/expression.csv')
    args.response_path = os.path.join(args.dataset_dir, f'data.csv')
    args.geneset_feature_path = os.path.join(args.dataset_dir, f'cell/geneset_exp')

    args.datasplit_path = os.path.join(args.dataset_dir, f'data_split')
    args.train = os.path.join(args.datasplit_path, f'Training.csv')
    args.valid = os.path.join(args.datasplit_path, f'Validation.csv')
    args.test = os.path.join(args.datasplit_path, f'Test.csv')

def get_prism_drug_path(args):
    args.drug_feat_types = args.drug_feat_type.split('+')  
    args.drug_path = dict()
    for drug_feat in args.drug_feat_types[1:]:
        args.drug_path[drug_feat] = get_single_prism_drug_path(args,drug_feat)

def get_single_prism_drug_path(args,sdrug_feat_type):
    if sdrug_feat_type in ['afp']:
        drug_path = os.path.join(args.dataset_dir, 'drug/drug.csv')
    elif sdrug_feat_type == 'kinome':
        drug_path = os.path.join(args.dataset_dir, 'drug/kinome/kinome.csv')
    elif sdrug_feat_type in ['GATtsall']:
        drug_path = os.path.join(args.dataset_dir, 'drug/transcriptome/transcriptall.csv')
    else:
        raise ValueError('drug feature type not supported')

    if sdrug_feat_type in ['GATtsall']:
        args.drug_ppi_path = os.path.join(args.dataset_dir, 'drug/transcriptome/gene_string_mapping.csv')
    return drug_path





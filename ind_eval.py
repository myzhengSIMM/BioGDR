# individual evaluation
import copy
import time
from configuration.path_config import get_path_config
from configuration.config import load_args,save_args
import os
from _utils.utils import makedirs
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler,SequentialSampler
from _utils.csv_dataset import ResponseDataset
from _utils.utils import get_collate_fn, log_judge, create_save_dir, redirect_log,drug_feat_to_device,get_data_path
from _utils.data_utils import feat_dict
from configuration.config import load_args,save_args,get_device_config
from _utils.model_utils import get_loss_func, build_optimizer
from _models.models import get_model
import torch
from evaluation import make_prediction
import numpy as np
from argparse import ArgumentParser
import pandas as pd
from torch import nn


def target_eval(args, feat_dict,logger=None):
    debug, info = log_judge(logger)
    test_dataset = ResponseDataset(args, args.target_dataset_path, feat_dict)  
    collate_fn = get_collate_fn(args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(test_dataset),   
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = get_model(args, feat_dict)

    args.drug_feat_func = drug_feat_to_device(args)

    args.loss_fn = get_loss_func(args)
    args.feat_dict = feat_dict

    debug(f'Moving model to device: {args.device}')
    model.load_state_dict(torch.load(args.best_model_path,map_location='cpu'))
    model = model.to(args.device,non_blocking=True)

    test_results, test_loss = make_prediction(args, model, test_loader, data_type='eval', save_preds=args.save_preds,
                                              get_attns=args.get_attn)


def parse_eval_args():
    parser = ArgumentParser()
    parser.add_argument('--cpu',action='store_true',default=False,help="Use cpu device")
    parser.add_argument('--gpu', type=int,default=2,choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--result_dir', type=str,default='data/train_example/results/train_example',
                        help='the result using for analysis')
    parser.add_argument('--target_data_dir', type=str,default='data/eval_example',
                        help='the result using for analysis')
    parser.add_argument('--save_dir',type=str,default=None,help='Specify the save path. If not set, defaults to the dataset directory.')    
    parser.add_argument('--num_workers', type=int, default=0,help='The number of workers')
    parser.add_argument('--get_attn',action='store_true',default=False,help = 'get attention score for gene and pathways')
    args = parser.parse_args()
    return args

def modify_ori_args_by_eval_args(ori_args,eval_args):
    #modifiy ori_args by eval_args
    ori_args.cpu = eval_args.cpu
    ori_args.gpu = eval_args.gpu
    ori_args.result_dir=eval_args.result_dir
    ori_args.get_attn=eval_args.get_attn


if __name__ == "__main__":

    # Get and modify evaluation parameters
    eval_args = parse_eval_args()
    target_data_dir=eval_args.target_data_dir  #dataset for evaluation
    args_path = os.path.join(eval_args.result_dir,'args.pickle')
    ori_args = load_args(args_path)
    ori_args.args_dir = args_path
    modify_ori_args_by_eval_args(ori_args,eval_args)

    #get path
    ori_args.dataset_dir = target_data_dir
    get_path_config(ori_args)   
    ori_args.target_dataset_path = os.path.join(target_data_dir,"data.csv")
    get_device_config(ori_args)

    #model_path
    ori_args.mode = 'eval'
    ori_args.fttaskid = time.strftime(f"%Y%m%d_%H%M%S",time.strptime(time.ctime(time.time()+8*60*60)))
    ori_args.best_model_path = os.path.join(eval_args.result_dir,'model','best_model_param.pickle')

    #save_path
    model_file_name = os.path.basename(eval_args.result_dir)
    if eval_args.save_dir is None:
        ori_args.save_dir =  os.path.join(target_data_dir,'results',model_file_name+"_"+ori_args.fttaskid) 
    else:
        ori_args.save_dir = eval_args.save_dir
    ori_args.args_dir = os.path.join(ori_args.save_dir,'args.pickle')

   #evaluation     
    every_feat_dict = feat_dict(ori_args)  
    makedirs(ori_args.save_dir)
    save_args(ori_args)
    logger = redirect_log(ori_args, False) 
    debug, info = log_judge(logger)
    target_eval(ori_args,every_feat_dict,logger)

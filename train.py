# -*- coding: utf-8 -*-
# Main file for model training
import copy
import sys
import time
from tqdm import tqdm
from torch.utils.data.sampler import RandomSampler
import torch
import os
from torch.utils.data import DataLoader
from argparse import Namespace
import pandas as pd
import numpy as np

from collections import defaultdict
from _utils.csv_dataset import ResponseDataset
from _utils.utils import  get_collate_fn, log_judge, create_save_dir,redirect_log,drug_feat_to_device
from _utils.data_utils import feat_dict
from _utils.metrics_utils import optimization_direction
from _utils.model_utils import get_loss_func, build_optimizer
from _models.models import get_model
from evaluation import make_prediction,eval,ind_eval
from configuration.config import get_config

# Main function for model training  
def train_main(args, use_feat_dict, logger=None):
    debug, info = log_judge(logger)
    debug('start training')
    debug('Loading data')
    train_dataset = ResponseDataset(args,args.train ,use_feat_dict, logger=logger)  
    val_dataset = ResponseDataset(args,args.valid ,use_feat_dict, logger=logger)

    collate_fn = get_collate_fn(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),     
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(val_dataset),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize model
    model = get_model(args, use_feat_dict,logger)
    args.drug_feat_func = drug_feat_to_device(args)

    # loss function and optimizer
    args.loss_fn = get_loss_func(args)
    args.optimizer = build_optimizer(model, args)

    debug(f'Moving model to device: {args.device}')
    model = model.to(args.device,non_blocking=True)

    # early stopping
    best_epoch = 0
    opt_d = optimization_direction[args.metric]
    if opt_d ==1 :
        best_val = -np.inf
    else:
        best_val = np.inf

    metric_dict = defaultdict(list)
    metrics_func = args.metric_func
    start_epoch = 0
    
    for epoch in tqdm(range(start_epoch, args.epochs), file=sys.stderr):
        # Train
        model.train()
        for batch_id, batch_data in enumerate(train_loader):
            drug_id, cell_id, drug_feats, CL_feats, pathways_g, labels = batch_data
            labels,  CL_feats,pathways_g =  labels.to(args.device,non_blocking=True),CL_feats.to(args.device,non_blocking=True),pathways_g.to(args.device,non_blocking=True)
            drug_feats = args.drug_feat_func(drug_feats,args.device)   
            prediction,_ = model(drug_feats, CL_feats,pathways_g) 
            loss = args.loss_fn(prediction,labels)
            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()

        args.feat_dict = use_feat_dict 
        epoch_val_results,epoch_val_loss = make_prediction(args, model, val_loader, data_type='val',save_preds=False)
        epoch_val_scores = eval(args,epoch_val_results, metrics_func,epoch_val_loss)

        epoch_train_results,epoch_train_loss = make_prediction(args,model,train_loader,data_type='train',save_preds=False)
        epoch_train_scores = eval(args,epoch_train_results, metrics_func,epoch_train_loss)

        if (opt_d * epoch_val_scores[args.metric] > opt_d * best_val): 
            best_val, best_train, best_epoch = epoch_val_scores[args.metric], epoch_train_scores[args.metric], epoch


        metric_dict['epoch'].append(epoch)
        metric_dict['best_epoch'].append(best_epoch)
        metric_dict[f'best_val_{args.metric}'].append(best_val)

        for metric in metrics_func:
            metric_dict['train_' + metric].append(epoch_train_scores[metric])
            metric_dict['val_' + metric].append(epoch_val_scores[metric])

        metric_csv = pd.DataFrame.from_dict(metric_dict).round(4)
        metric_csv.to_csv(os.path.join(args.save_dir, f'metrics.csv'), index=False)


        if epoch - best_epoch > args.early_stop_epoch:
            break

    torch.save(model.state_dict(),os.path.join(args.model_dir, f"best_model_param.pickle"))
    args.best_epoch = best_epoch
    ind_eval(args,use_feat_dict,model)


if __name__ == "__main__":
    args = get_config()
    logger = redirect_log(args,False)   
    create_save_dir(args)
    use_feat_dict = feat_dict(args,logger)  
    train_main(args, use_feat_dict,logger)


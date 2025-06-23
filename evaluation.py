# evaluation function
import os
import torch
from collections import OrderedDict
from typing import List
import numpy as np
import pandas as pd
from _utils.csv_dataset import ResponseDataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
from _utils.utils import  get_collate_fn,scoring


def make_prediction(args, model, data, data_type, save_preds=False,get_attns = False):
    drug_ids, cell_ids, labels_np, preds_np ,g_attns,p_attns = [], [], [], [],[],[]
    eval_losses = 0

    with torch.no_grad():
        model.eval()
        for batch_id, batch_data in enumerate(data): 
            drug_id, cell_id, drug_feats, CL_feats, pathways_g, labels = batch_data
            labels, CL_feats, pathways_g = labels.to(args.device,non_blocking=True), CL_feats.to(
                args.device,non_blocking=True), pathways_g.to(args.device,non_blocking=True)
            drug_feats = args.drug_feat_func(drug_feats,args.device)  

            prediction,attns = model(drug_feats, CL_feats, pathways_g) 

            eval_loss = args.loss_fn(prediction, labels)  
            eval_losses += eval_loss.item()

            drug_ids.extend(drug_id)
            cell_ids.extend(cell_id)

            labels_np.append(labels.detach())  
            preds_np.append(prediction.detach()) 
            if get_attns:  
                g_attns.append(attns[0].detach().cpu().numpy()) 
                p_attns.append(attns[1].detach().cpu().numpy())  

        labels_np = torch.concat(labels_np,dim=0).cpu().numpy()       
        preds_np = torch.concat(preds_np,dim=0).cpu().numpy()    

        metric_loss = eval_losses / (batch_id + 1)

        labels_np = np.squeeze(labels_np)
        preds_np = np.squeeze(preds_np)

        results = pd.DataFrame(list(zip(drug_ids, cell_ids, labels_np, preds_np)),
                               columns=['drug_id', 'cell_id', 'label', 'prediction'])

        if get_attns:
            results_col = [] 
            for i in results.columns:
                results_col.append(("results",i))
            results.columns = pd.MultiIndex.from_tuples(results_col)

            g_col1,g_col2,p_col1,p_col2 = [],[],[],[]
            for k, v in args.feat_dict.GeneSet_Dict.items():
                p_col1.append("pathway")
                p_col2.append(k)
                for gene in v:
                    g_col1.append(k)
                    g_col2.append(gene)
            g_col = pd.MultiIndex.from_arrays(np.array([g_col1, g_col2]))
            p_col = pd.MultiIndex.from_arrays(np.array([p_col1,p_col2]))
  
            g_attns = pd.DataFrame(np.concatenate(g_attns, axis=0),columns=g_col)    
            p_attns = pd.DataFrame(np.concatenate(p_attns, axis=0),columns=p_col)   

            results = pd.concat([results,g_attns,p_attns],axis=1)

        if save_preds:
            results.to_csv(os.path.join(args.save_dir, f'{data_type}_results.csv'), index=False)

        return results, metric_loss

def eval(args, results, metrics_func='default', loss=None):
    if metrics_func != 'default': 
        if not isinstance(metrics_func, List):  
            metrics_func = [metrics_func]
        metrics_func = list(set(metrics_func)) 

    scores = OrderedDict()
    if 'loss' in metrics_func: 
        scores['loss'] = loss
        metrics_func.remove('loss')

    if 'label' not in results.columns:
        results = results['results']

    labels_np_sq = np.array(results['label'])
    preds_np_sq = np.array(results['prediction'])
    scores.update(scoring(labels_np_sq, preds_np_sq, dataset_type=args.dataset_type,
                          metrics_func=metrics_func)) 

    return scores

def ind_eval(args, feat_dict, model):
    train_dataset = ResponseDataset(args, args.train, feat_dict)
    val_dataset = ResponseDataset(args, args.valid, feat_dict)
    test_dataset = ResponseDataset(args, args.test, feat_dict)

    collate_fn = get_collate_fn(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset), 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(val_dataset),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(test_dataset),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    best_model_param_path = os.path.join(args.model_dir, f"best_model_param.pickle")
    model.load_state_dict(torch.load(best_model_param_path))
    args.feat_dict = feat_dict

    train_results, train_loss = make_prediction(args, model, train_loader, data_type='train',save_preds=args.save_preds,get_attns=args.get_attn) 
    val_results, val_loss= make_prediction(args, model, val_loader, data_type='val', save_preds=args.save_preds,get_attns=args.get_attn)
    test_results, test_loss = make_prediction(args, model, test_loader, data_type='test', save_preds=args.save_preds,get_attns=args.get_attn)

    metrics_func = 'default'

    overall = OrderedDict()

    overall['train'] = eval(args, train_results, metrics_func, train_loss)
    overall['val'] = eval(args, val_results, metrics_func, val_loss)
    overall['test'] = eval(args, test_results, metrics_func, test_loss)

    overall_df = pd.DataFrame.from_dict(overall).transpose().round(4)
    overall_df.to_csv(os.path.join(args.save_dir, f'final_metrics.csv'))

    return overall['test']



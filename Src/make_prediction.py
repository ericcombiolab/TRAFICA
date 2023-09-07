from transformers import BertConfig, BertTokenizer
from SeqFolder import SeqFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import sys
import sklearn.metrics as metrics
from scipy.stats import pearsonr
import argparse
import json
from models import Bert_seqRegression_adapterfusion, Bert_seqRegression, Bert_seqRegression_adapter
import scipy.stats as stats

sys.path.append('..')
from utils import *



def data_collate_fn(arr):   # padding with [PAD] / tokenizing bases into numbers when loading sequence data
    collect_x = []
    collect_y = []
    for k_mer in arr:
        tmp =k_mer[:-1]
        y = k_mer[-1]
        collect_x.append(' '.join(tmp))
        collect_y.append(float(y))
        #collect_y.append(int(y))
    return collect_x, collect_y
    #return Tokenizer(collect, return_tensors="pt", padding=True), collect_y


def load_datasets_test(data_path, batch_size, k=4):
    data = SeqFolder(data_path)

    dataloader = DataLoader(data, batch_size, shuffle=False, collate_fn=data_collate_fn)
    return dataloader


def test_model(model, test_data, Tokenizer, device):
    model.eval()
    collect_y = []
    collect_out = []
    for data in test_data:
        X, y = data
        inputs = Tokenizer(X, return_tensors="pt", padding=True)
        y = torch.tensor(data[1]).float()
        out = model(inputs.to(device))

        collect_y += y.tolist()
        collect_out += out.flatten().cpu().detach().numpy().tolist()

    return  collect_y, collect_out





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=False, type=int, default=1, help="Batch size for evaluation. Default:1")
    parser.add_argument('--use_gpu', default=False, type=bool, help="Turn on to use gpu to train model; False: using cpu, True: using gpu if available.")
    parser.add_argument('--save_dir', required=True,type=str, help="The saving path of the trained model")
    parser.add_argument('--data_path', required=True, type=str, help="The path of the data folder")
    parser.add_argument('--vocab_path', required=True, type=str, help="The path of the vocabulary")
    parser.add_argument('--model_dir', required=True, type=str, help="The path of the finetuned model.")    
    parser.add_argument('--pretrained_model_path', required=False, type=str, help="The path of the pretrained model")
    parser.add_argument('--pretrained_adapters_path', required=False, type=str, help="The path of the pretrained adapters")

    parser.add_argument('--k', required=False, default=4, type=int, help="K-mer token hyper-parameter.")

    parser.add_argument('--task_type', default='classification', type=str, help=".")
    parser.add_argument('--pool_strategy', default='t_cls', type=str, help="Use [CLS] embedding or average all k-mer embeddings as sequence-level representation.")
    parser.add_argument('--evaluation_metric', required=True, default='set1', type=str, help="set1: auroc; set2: pcc & r2")

    args = parser.parse_args()



    set_seeds(3047)  # torch.manual_seed(3407) is all you need

    # test device
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'


    # model saving
    create_directory(args.save_dir)
    

    # load pre-trained model & tokenizer initialization
    Tokenizer = BertTokenizer(args.vocab_path, do_lower_case=False, model_max_length=512)

    # datasets of each experiment
    data = load_datasets_test(args.data_path, batch_size=args.batch_size, k=args.k)
    
      
    if args.task_type == 'regression': # load fine-tuned model
        configuration = BertConfig.from_pretrained(args.model_dir)  
        model = Bert_seqRegression(configuration, model_path=args.model_dir, pool_strategy=args.pool_strategy, fine_tuned=True)
    elif args.task_type == 'regression_adapter':    # load pre-trained model and trained adapter
        model = Bert_seqRegression_adapter(pretrain_modelpath=args.pretrained_model_path, model_path=args.model_dir, pool_strategy=args.pool_strategy, fine_tuned=True)   
    elif args.task_type == 'regression_adapterfusion': # load pre-trained model, trained adapters, and adapterfusion
        with open(args.pretrained_adapters_path, "r") as f:
            adapters = json.load(f)
        model = Bert_seqRegression_adapterfusion(pretrain_modelpath=args.pretrained_model_path, pretrain_adapterpath=adapters, model_path=args.model_dir, pool_strategy=args.pool_strategy, fine_tuned=True, out_attention=False)   
        


    if torch.cuda.is_available():
        model.to(device)



    test_y, test_y_hat = test_model(model, test_data=data, Tokenizer=Tokenizer, device=device)
    

    # saving test result
    save_txt_single_column(os.path.join(args.save_dir, 'test_y.txt'), test_y)
    save_txt_single_column(os.path.join(args.save_dir, 'test_y_hat.txt'), test_y_hat)


    if args.evaluation_metric == 'set1':
        auroc = metrics.roc_auc_score(y_true=test_y, y_score=test_y_hat)
        save_txt_single_column(os.path.join(args.save_dir, 'auroc.txt'), auroc)

    elif args.evaluation_metric == 'set2':
        r2 = metrics.r2_score(test_y, test_y_hat)
        pcc, _ = pearsonr(test_y, test_y_hat)
        save_txt_single_column(os.path.join(args.save_dir, 'r2.txt'), r2)
        save_txt_single_column(os.path.join(args.save_dir, 'pcc.txt'), pcc)


# CUDA_VISIBLE_DEVICES=3 python make_prediction.py \
# --data_path /tmp/csyuxu/chip_seq_datasets_4mer_DataFolder/CEBPB_K562_CEBPB_Stanford/test.4mer \
# --model_dir /tmp/csyuxu/finetuned_adapter_fusion/PRJEB14744_CEBPB_TGGACA20NGA \
# --save_dir ../tmp_test \
# --vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
# --use_gpu True \
# --pretrained_model_path ./pretrained_models/ATAC_seq_10split_4mer \
# --pool_strategy mean \
# --task_type regression_adapterfusion \
# --evaluation_metric set1 \
# --pretrained_adapters_path ./adapter_maps/PRJEB14744_CEBPB_TGGACA20NGA.json \
# --batch_size 2


# CUDA_VISIBLE_DEVICES=0 python make_prediction.py \
# --data_path /tmp/csyuxu/chip_seq_datasets_4mer_DataFolder/CEBPB_H1-hESC_CEBPB_Stanford/test.4mer \
# --model_dir /tmp/csyuxu/finetuned_adapter_fusion/PRJEB14744_CEBPB_TGGACA20NGA \
# --save_dir ../tmp_test \
# --vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
# --use_gpu True \
# --pretrained_model_path ./pretrained_models/ATAC_seq_10split_4mer \
# --pool_strategy mean \
# --task_type regression_adapterfusion \
# --evaluation_metric set1 \
# --pretrained_adapters_path ./adapter_maps/PRJEB14744_CEBPB_TGGACA20NGA.json \
# --batch_size 2


# CUDA_VISIBLE_DEVICES=3 python make_prediction.py \
# --data_path /tmp/csyuxu/chip_seq_datasets_4mer_DataFolder/CEBPB_H1-hESC_CEBPB_Stanford/test.4mer \
# --model_dir ./PRJEB14744_finetuned_adapter/CEBPB_TGGACA20NGA \
# --save_dir ../tmp_test \
# --vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
# --use_gpu True \
# --pretrained_model_path ./pretrained_models/ATAC_seq_10split_4mer \
# --pool_strategy mean \
# --task_type regression_adapter \
# --evaluation_metric set1 \
# --batch_size 32

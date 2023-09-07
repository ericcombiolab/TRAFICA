import os
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import json
import sys

sys.path.append('..')
from utils import *


def escape_parentheses(filepath):
    """
    This function takes a file path as input and returns a modified file path with escaped characters for parentheses.
    """
    return filepath.replace('(', '\(').replace(')', '\)')

def get_direct_parent_directory(filepath):
    """
    This function takes a file path as input and returns the direct parent directory of the file.
    """
    return os.path.basename(os.path.dirname(filepath))



# basic params
cuda_device = 2
use_gpu = True
vocab_path = '../vocab_k_mer/vocab_DNA_4_mer.txt'
task_type = 'regression_adapter' # 'regression_adapterfusion'
batch_size = 32

# load evaluation pairs
pairs_info_path = '../HTSELEX_CHIP_overlap.csv'
df_pairs = pd.read_csv(pairs_info_path)

# path
adapterfusion_dir = './finetuned_adapter_fusion'
adapter_maps_dir = './adapter_maps'
pretrained_model_path = './pretrained_models/ATAC_seq_10split_4mer'
data_dir = '/tmp/csyuxu/chip_seq_datasets_4mer_DataFolder'
save_dir = '/tmp/csyuxu/HTSELEX_CHIP_evaluation'


evaluation_metric = 'set1'


for _, item in df_pairs.iterrows():
    print('processing:\t', _ )

    htselex_study = item['study']
    htselex_key = item['htselex_key']
    chip_key = item['chip_key']
  
    model_key = htselex_study + '_' + htselex_key
    if task_type == 'regression_adapterfusion':
        model_path = os.path.join(adapterfusion_dir, model_key)
    elif task_type == 'regression_adapter':
        model_path = os.path.join(htselex_study+'_finetuned_adapter', htselex_key)

    adapter_path = os.path.join(adapter_maps_dir, model_key +'.json')
    data_path = os.path.join(data_dir, chip_key, 'test.4mer')
    data_path = escape_parentheses(data_path)

    save_path = os.path.join(save_dir, model_key + '_' + chip_key )
    save_path = escape_parentheses(save_path)
    
   
    invivo_inference_params = f"CUDA_VISIBLE_DEVICES={cuda_device} python make_prediction.py " \
                    f"--model_dir {model_path} " \
                    f"--save_dir {save_path} " \
                    f"--data_path {data_path} " \
                    f"--vocab_path {vocab_path} " \
                    f"--use_gpu {use_gpu} " \
                    f"--pretrained_model_path {pretrained_model_path} " \
                    f"--task_type {task_type} " \
                    f"--pool_strategy mean " \
                    f"--evaluation_metric {evaluation_metric} " \
                    f"--pretrained_adapters_path {adapter_path} " \
                    f"--batch_size {batch_size} " 



    status = os.system(invivo_inference_params)
    if status != 0:
        break

    

# recording evaluation metrics of all pairs
evaluation_results = os.listdir(save_dir)
collect = {}
for i in evaluation_results:
    if i.split('_')[0] == 'test':
        continue

    if  evaluation_metric == 'set1':
        auroc = pd.read_table(os.path.join(save_dir, i, 'auroc.txt'),header=None).values.ravel()[0]
        collect[i] = auroc


if  evaluation_metric == 'set1':
    with open(os.path.join(save_dir, "test_auc.json"), "w") as write_file:
        json.dump(collect, write_file, indent=4)
## TODO: support other metrics 

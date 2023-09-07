import sys
from transfer_server import Transmit_Server
from parallel_recorder import Parallel_Recorder
import re


sys.path.append('..')
from utils import *



def escape_special_chars(s):
    pattern = r'([\(\)])'
    repl = r'\\\1'
    return re.sub(pattern, repl, s)


def get_finetune_data_BATCH_EXP(data_path):
    exps_dir = os.listdir(data_path)
    n_exps = len(exps_dir)
    exps_dir.sort()
    return exps_dir, n_exps


if __name__ == '__main__':

    cuda_device = 0
    batch_size = 8 # 128 for fine-tuning; 8 for adapter fusion
    n_epoches = 50
    n_epoches_tolerance = 3
    lr = 0.0001 ## 0.0001 for classification ; 0.00002 for regression by default; 0.0001 for adapter fusion
    k = 4
    vocab_path = "/home/comp/csyuxu/aptdrug/vocab_k_mer/vocab_DNA_4_mer.txt" 
    pretrained_model_path = "./pretrained_models/ATAC_seq_10split_4mer"
    use_gpu = True 
    task_type = "regression_adapterfusion" #"regression_adapter" # "classification_adapter" #"regression" # "regression_adapterfusion"
    pool_strategy = "mean"  #t_cls mean

    # these params only are used in adapter fusion
    max_num_trainsample = 0 
    max_num_valsample = 0
    pretrained_adapters_path = 0

    ### PBM tuning
    # save_dir_metric = "/tmp/csyuxu/DREAM5_PBM_finetuned"
    # batch_data_path = "/tmp/csyuxu/DREAM5_PBM_protocol_4mer_DataFolder" 
    # batch_save_dir = "/tmp/csyuxu/DREAM5_PBM_finetuned"
    # record_file = "DREAM5_PBM.txt"
    # #remote_path = None
    # remote_path = "/datahome/datasets/ericteam/csyuxu/PBM_finetuned_models/DREAM5"

    ### HT-SELEX tuning
    # save_dir_metric = "./PRJEB14744_finetuned_adapter"
    # batch_data_path = "/tmp/csyuxu/PRJEB14744_4mer_finetuneDataFolder" 
    # batch_save_dir = "./PRJEB14744_finetuned_adapter"
    # record_file = "PRJEB14744_adapter.txt"
    # remote_path = None

    # ### HT-SELEX full-tuning
    # save_dir_metric = "/tmp/csyuxu/PRJEB9797_PRJEB20112_finetuned"
    # batch_data_path = "/tmp/csyuxu/PRJEB9797_PRJEB20112_4mer_finetuneDataFolder" 
    # batch_save_dir = "/tmp/csyuxu/PRJEB9797_PRJEB20112_finetuned"
    # record_file = "PRJEB9797_PRJEB20112.txt"
    # remote_path = '/datahome/datasets/ericteam/csyuxu/HTSELEX_finetuned_models/PRJEB9797_PRJEB20112_FT'


    ### ChIP-seq tuning
    # save_dir_metric = "./ChIP_seq_finetuned_adapter"
    # batch_data_path = "/tmp/csyuxu/processed_ChIP_seq_DeepBind_4mer_finetuneDataFolder" 
    # batch_save_dir = "./ChIP_seq_finetuned_adapter"
    # record_file = "ChIP_seq_adapter.txt"
    # remote_path = None

    ### adapter fusion
    save_dir_metric = "/tmp/csyuxu/finetuned_adapter_fusion"
    batch_data_path = "/tmp/csyuxu/htselex_datasets_4mer_DataFolder" 
    batch_save_dir = "/tmp/csyuxu/finetuned_adapter_fusion"
    record_file = "adapter_fusion.txt"
    remote_path = None
    max_num_trainsample = 2000 
    max_num_valsample = 1000
    pretrained_adapters_path = './adapter_maps'
       
    # load fine-tune data
    exps, n_exps = get_finetune_data_BATCH_EXP(batch_data_path)

    ## dynamic parallel running and saving results
    recorder = Parallel_Recorder(record_file=record_file, task_list=exps)

    ## send results to other server    
    if remote_path != None:
        tranfer_station = Transmit_Server(pass_word='000000', save_folder=batch_save_dir,target_path=remote_path)

    ## bacth fine-tune and evaluate models
    while len(recorder.unfinished_exps) != 0:    
        # add this exp into finished list
        current_exp = recorder.unfinished_exps[0]
        recorder.update(current_exp)

        name_experiment = current_exp
        save_dir = os.path.join(batch_save_dir, current_exp)
        data_path = os.path.join(batch_data_path, current_exp)

        finetuning_params = f"CUDA_VISIBLE_DEVICES={cuda_device} python finetuning.py " \
                        f"--batch_size {batch_size} " \
                        f"--n_epoches {n_epoches} " \
                        f"--n_epoches_tolerance {n_epoches_tolerance} " \
                        f"--lr {lr} --k {k} " \
                        f"--save_dir {save_dir} " \
                        f"--data_path {data_path} " \
                        f"--vocab_path {vocab_path} " \
                        f"--pretrained_model_path {pretrained_model_path} " \
                        f"--use_gpu {use_gpu} " \
                        f"--task_type {task_type} " \
                        f"--pool_strategy {pool_strategy} " \
                        f"--name_experiment {name_experiment} " \
                        f"--save_dir_metric {save_dir_metric}"

        if task_type == "regression_adapterfusion":
            pretrained_adapter = os.path.join(pretrained_adapters_path, current_exp+'.json')
            addition_params = f"--pretrained_adapters_path {pretrained_adapter} " \
                              f"--max_num_trainsample {max_num_trainsample} " \
                              f"--max_num_valsample {max_num_valsample}"

            finetuning_params = finetuning_params + ' ' + addition_params

      

        status = os.system(escape_special_chars(finetuning_params))
        if status!=0:
            print("error in fune-tuning:\t", current_exp)
            break

        # send the results to data server 
        if remote_path != None:
            tranfer_station.step(current_exp)
            if task_type in ['classification', 'classification_adapter']:
                for metric in ['auroc', 'auprc']:
                    tranfer_station.send_one_file(os.path.join(batch_save_dir, 'test_' + metric +'.json'))
            elif task_type in ['regression', 'regression_adapter']: 
                for metric in ['r2', 'pcc', 'mse']:
                    tranfer_station.send_one_file(os.path.join(batch_save_dir, 'test_' + metric +'.json'))
 
        # check unfinished experiments
        recorder.refresh()      
   



import os
import pandas as pd
import argparse

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()


    parser.add_argument('--mode', required=True, default='HT-SELEX', type=str, help=".")
    parser.add_argument('--tokenization', required=False, default='Base-level', type=str, help=".")

    args = parser.parse_args()
    
    
    
    mode = args.mode                    # PBM; HT-SELEX ; Ablation_PreTrain
    tokenization = args.tokenization    # 4_mer; 5_mer; 6_mer; Base-level; BPE; BPE_DNABERT
    
    if tokenization == 'BPE_DNABERT':
        path_tokenizer = 'zhihan1996/DNABERT-2-117M'
    else:
        path_tokenizer = os.path.join('./Tokenizers', tokenization)
 
    # path_tokenizer = './Tokenizers/Base-level'
    # tokenization = 'Base-level'

    # path_tokenizer = './Tokenizers/5_mer'
    # tokenization = '5_mer'

    # path_tokenizer = './Tokenizers/BPE'
    # tokenization = 'BPE'

    # path_tokenizer = 'zhihan1996/DNABERT-2-117M'
    # tokenization = 'BPE_DNABERT'

    if mode == 'Cross-platform':
        ### PBM
        df = pd.read_csv('../Data/HT_SELEX_PBM_DREAM5_overlap.csv')
        for num_seqs in [10000,20000,30000,40000,50000]:
            for _, item in df.iterrows():
                ht_selex_study = item['htselex_study']
                ht_selex_key = item['htselex_key']
                pbm_key = item['pbm_key']

                eval_data_path = os.path.join('../Data/2_Eval_HTSELEX_PBM', f"{pbm_key}.txt")
                save_dir = os.path.join(f'../Eval_TRAFICA/HTSELEX_PBM_Evaluation/{tokenization}/{num_seqs}', f"{ht_selex_study}_{ht_selex_key}_{pbm_key}")

                if os.path.exists(os.path.join(save_dir, 'pcc.txt')):
                    continue

                command = f"CUDA_VISIBLE_DEVICES=1 python eval_TFbinding.py " \
                            f"--batch_size {128} " \
                            f"--save_dir {save_dir} " \
                            f"--eval_data_path {eval_data_path} " \
                            f"--tokenizer_path {path_tokenizer} " \
                            f"--tokenization {tokenization} " \
                            f"--use_gpu {True} " \
                            f"--predict_type regression " \
                            f"--pretrained_model_path ../Pretrained_TRAFICA/medium_{tokenization}/TRAFICA_Weights " \
                            f"--finetuned_lora_path ../Finetuned_TRAFICA_All/{tokenization}/{ht_selex_study}/{num_seqs}/{ht_selex_key}" 
                                
                print(command)
                code = os.system(command)
                # break

    elif mode == 'In-vivo':
        ### ChIP-seq
        df = pd.read_csv('../Data/HT_SELEX_ChIP_overlap.csv')

        for num_seqs in [10000,20000,30000,40000,50000]:
            for _, item in df.iterrows():
                ht_selex_study = item['ht_selex_study']
                ht_selex_key = item['ht_selex_key']
                chip_key = item['chip_key']
                
                
                eval_data_path = os.path.join('../Data/3_Eval_HTSELEX_ChIP', f"{chip_key}.txt")
                save_dir = os.path.join(f'../Eval_TRAFICA/HTSELEX_ChIP_Evaluation/{tokenization}/{num_seqs}', f"{ht_selex_study}_{ht_selex_key}_{chip_key}")

                if os.path.exists(os.path.join(save_dir, 'auroc.txt')):
                    continue

                command = f"CUDA_VISIBLE_DEVICES=0 python eval_TFbinding.py " \
                            f"--batch_size {128} " \
                            f"--save_dir {save_dir} " \
                            f"--eval_data_path {eval_data_path} " \
                            f"--tokenizer_path {path_tokenizer} " \
                            f"--tokenization {tokenization} " \
                            f"--use_gpu {True} " \
                            f"--predict_type classification " \
                            f"--pretrained_model_path ../Pretrained_TRAFICA/medium_{tokenization}/TRAFICA_Weights " \
                            f"--finetuned_lora_path ../Finetuned_TRAFICA_All/{tokenization}/{ht_selex_study}/{num_seqs}/{ht_selex_key}" 
                                
                print(command)
                code = os.system(command)
                # break


    elif mode == 'Cross-experiment':
        ##### HT-SELEX cross-exp
        df = pd.read_csv('../Data/HT_SELEX_CrossExp_overlap.csv')

        for num_seqs in [10000,20000,30000,40000,50000]:
            for _, item in df.iterrows():
                train_study = item['train_study']
                train_key = item['train_key']
                test_study = item['test_study']
                test_key = item['test_key']
                
                
                eval_data_path = os.path.join('../Data/HT_SELEX', test_study, str(num_seqs), test_key, "test.txt")
                save_dir = os.path.join(f'../Eval_TRAFICA/HTSELEX_CrossExp_Evaluation/{tokenization}/{num_seqs}', f"{train_study}_{train_key}_{test_study}_{test_key}")

                if os.path.exists(os.path.join(save_dir, 'pcc.txt')):
                    continue

                command = f"CUDA_VISIBLE_DEVICES=0 python eval_TFbinding.py " \
                            f"--batch_size {128} " \
                            f"--save_dir {save_dir} " \
                            f"--eval_data_path {eval_data_path} " \
                            f"--tokenizer_path {path_tokenizer} " \
                            f"--tokenization {tokenization} " \
                            f"--use_gpu {True} " \
                            f"--predict_type regression " \
                            f"--pretrained_model_path ../Pretrained_TRAFICA/medium_{tokenization}/TRAFICA_Weights " \
                            f"--finetuned_lora_path ../Finetuned_TRAFICA_All/{tokenization}/{train_study}/{num_seqs}/{train_key}" 
                                
                print(command)
                code = os.system(command)
                # break


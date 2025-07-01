import os
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


    if  mode == 'PBM':
        ##### DREAM5
        folder = os.path.join('../Data/DREAM5_PBM_protocol')
        PBM_Exp = os.listdir(folder)
        for exp in PBM_Exp:
            eval_data_path = os.path.join(folder, exp)
            save_dir = os.path.join(f'../Finetuned_TRAFICA_All/{tokenization}', 'FT_PBM', exp)
            
            if os.path.exists( os.path.join(save_dir, 'pcc.txt')):
                continue
            
            command = f"CUDA_VISIBLE_DEVICES=3 python finetuning_TFbinding.py " \
                        f"--batch_size {128} " \
                        f"--n_epoch {300} " \
                        f"--n_toler {10} " \
                        f"--lr {0.0001} " \
                        f"--save_dir {save_dir} " \
                        f"--eval_data_path {eval_data_path} " \
                        f"--tokenizer_path {path_tokenizer} " \
                        f"--tokenization {tokenization} " \
                        f"--use_gpu {True} " \
                        f"--predict_type regression " \
                        f"--pretrained_model_path ../Pretrained_TRAFICA/medium_{tokenization}/TRAFICA_Weights"
            
            code = os.system(command)
            
            
    elif mode == 'HT-SELEX':
        ##### HT-SELEX subset
        for study in ['PRJEB14744','PRJEB3289','PRJEB9797_PRJEB20112']:
            for num_seqs in [10000,20000,30000,40000,50000]:
                folder = os.path.join('../Data/HT_SELEX', study, str(num_seqs))
                HT_SELEX_Exp = os.listdir(folder)
                for exp in HT_SELEX_Exp:
                    eval_data_path = os.path.join(folder, exp)
                    save_dir = os.path.join(f'../Finetuned_TRAFICA_All/{tokenization}', study, str(num_seqs), exp)
                    
                    
                    # check = save_dir                                # first running
                    check = os.path.join(save_dir, 'pcc.txt')     # check missing 
                
                    if os.path.exists(check):
                        continue
                    
                    command = f"CUDA_VISIBLE_DEVICES=1 python finetuning_TFbinding.py " \
                                f"--batch_size {128} " \
                                f"--n_epoch {300} " \
                                f"--n_toler {10} " \
                                f"--lr {0.0001} " \
                                f"--save_dir {save_dir} " \
                                f"--eval_data_path {eval_data_path} " \
                                f"--tokenizer_path {path_tokenizer} " \
                                f"--tokenization {tokenization} " \
                                f"--use_gpu {True} " \
                                f"--predict_type regression " \
                                f"--pretrained_model_path ../Pretrained_TRAFICA/medium_{tokenization}/TRAFICA_Weights"

                    print(command)
                    code = os.system(command)



    elif mode == 'Ablation_PreTrain':
        ##### DREAM5 without pre-training
        folder = os.path.join('../Data/DREAM5_PBM_protocol')
        PBM_Exp = os.listdir(folder)
        # mode_ = 'train_from_scratch' # 'pre_trained' 'train_from_scratch'
        for mode_ in [ 'pre_trained', 'train_from_scratch']:
            for exp in PBM_Exp:
                eval_data_path = os.path.join(folder, exp)
                save_dir = os.path.join(f'../AblationPreTrain_TRAFICA/{tokenization}_{mode_}', 'FT_PBM', exp)
                
                if os.path.exists( os.path.join(save_dir, 'pcc.txt')):
                    continue
                
                command = f"CUDA_VISIBLE_DEVICES=0 python trainingfromscratch_TFbinding.py " \
                            f"--mode {mode_} " \
                            f"--batch_size {128} " \
                            f"--n_epoch {300} " \
                            f"--n_toler {10} " \
                            f"--lr {0.0001} " \
                            f"--save_dir {save_dir} " \
                            f"--eval_data_path {eval_data_path} " \
                            f"--tokenizer_path {path_tokenizer} " \
                            f"--tokenization {tokenization} " \
                            f"--use_gpu {True} " \
                            f"--predict_type regression " \
                            f"--pretrained_model_path ../Pretrained_TRAFICA/medium_{tokenization}/TRAFICA_Weights"
                
                print(command)
                code = os.system(command)
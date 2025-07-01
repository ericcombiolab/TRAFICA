import os


path_tokenizer = './Tokenizers/Base-level'
tokenization = 'Base-level'

path = '../Finetuned_TRAFICA_All/Base-level'
data_path = '../Data/HT_SELEX'
for study in ['PRJEB14744','PRJEB3289','PRJEB9797_PRJEB20112']:
    folder = os.path.join(path, study, str(50000))
    Study_Exp = os.listdir(folder)
    for exp in Study_Exp:
        finetuned_lora_path = os.path.join(folder, exp)
        eval_data_path = os.path.join(data_path, study, str(50000), exp, 'test.txt')
        
        save_dir = os.path.join(f'../Motif_Attn/{tokenization}', study, str(50000), exp, 'Attentions')

        
        command = f"CUDA_VISIBLE_DEVICES=1 python attn_extraction.py " \
                    f"--batch_size {128} " \
                    f"--save_dir {save_dir} " \
                    f"--eval_data_path {eval_data_path} " \
                    f"--tokenizer_path {path_tokenizer} " \
                    f"--tokenization {tokenization} " \
                    f"--use_gpu {True} " \
                    f"--finetuned_lora_path {finetuned_lora_path} " \
                    f"--pretrained_model_path ../Pretrained_TRAFICA/medium_{tokenization}/TRAFICA_Weights"
        print( command )
        code = os.system(command)
    #     break
    # break  
    

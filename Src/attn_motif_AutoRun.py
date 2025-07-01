import os


attn_path = '../Motif_Attn/Base-level'
for study in ['PRJEB14744','PRJEB3289','PRJEB9797_PRJEB20112']:
    folder = os.path.join(attn_path, study, str(50000))
    Study_Exp = os.listdir(folder)
    for exp in Study_Exp:
        path = os.path.join(folder, exp)

        attn_data_path = os.path.join(path, 'Attentions','attns.npz')
        
        command = f"CUDA_VISIBLE_DEVICES=1 python attn_motif.py " \
                    f"--path {path} " \
                    f"--attn_data_path {attn_data_path} " 
        print( command )
        code = os.system(command)
    #     break
    # break  
    
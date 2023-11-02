import os
import pandas as pd
import json
import sys

sys.path.append('..')
from utils import *

# HT-SELEX 1 adapter + ChIP-seq 137 adapters
if True: 
    save_dir = '../AdapterMapsForFusion/ChIP_seq_137'
    create_directory(save_dir)
    chip_adapter = os.listdir('../Adapters/TF_DNA_Adapters/ChIP_seq')
    df = pd.read_csv('../HTSELEX_CHIP_overlap.csv')


    repeat_chip_tf = []
    for _, item in df.iterrows():
        adapter_collect = {}
        study = item['study']
        htselex_key = item['htselex_key']

        ## avoid 1 tf map to different chip-seq evaluation dataset -> make crash files with same content
        chip_key = item['chip_key']
        chip_tf_current_row = chip_key.split('_')[0]
        if chip_tf_current_row not in repeat_chip_tf:
            repeat_chip_tf.append(chip_key)
        else:
            continue
        map_filename = study + '_' + htselex_key + '.json'
        
        ## main task adapter 
        adapter_collect[study + '_' + htselex_key] = os.path.join('../Adapters/TF_DNA_Adapters', study, htselex_key)

        target_tf, _ = htselex_key.split('_')

        repeat_tf_cellline = []
        for adapter in chip_adapter:
            chip_tf = adapter.split('_')[0]

            if chip_tf not in repeat_tf_cellline:
                repeat_tf_cellline.append(chip_tf)
            else:
                continue
    

            if chip_tf == 'test': # other files, ignore it
                continue
            ## do not use the chip-seq information from the same tf
            if target_tf == chip_tf:   
                continue
        
            adapter_collect[adapter] = os.path.join('../Adapters/TF_DNA_Adapters/ChIP_seq', adapter)

        with open(os.path.join(save_dir, map_filename), 'w') as json_file:
            json.dump(adapter_collect, json_file,indent=4, separators=(',', ': '))

    # break
            

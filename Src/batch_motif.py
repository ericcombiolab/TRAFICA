import os
import re



if __name__ == '__main__':


    finetuned_folder = '/tmp/csyuxu/HTSELEX_finetuned_models'
    study = 'PRJEB3289'  # PRJEB14744, PRJEB9797_PRJEB20112
    study_folder = os.path.join(finetuned_folder, study)
    
    experiments = os.listdir(study_folder)
   
    for experiment_key in experiments:
        model_dir = os.path.join(study_folder, experiment_key)
        if not os.path.isdir(model_dir):
            continue

        print('Processing', experiment_key)

 
        save_dir = os.path.join('/tmp/csyuxu/motif_save', study, experiment_key)

        min_length = 4 


        mode_obtain_attScore = True
        n_seqs = 1000   
        data_path = os.path.join('/tmp/csyuxu/Data_forMotif',study, experiment_key+'.txt')
        vocab_path = '../vocab_k_mer/vocab_DNA_4_mer.txt'
        use_gpu = True

        ### running this to calculate attention score only
        # motif_params =  f"python motif.py " \
        #                 f"--calc_score {mode_obtain_attScore} " \
        #                 f"--n_sequences {n_seqs} " \
        #                 f"--data_path {data_path} " \
        #                 f"--model_dir {model_dir} " \
        #                 f"--save_dir {save_dir} " \
        #                 f"--vocab_path {vocab_path} " \
        #                 f"--use_gpu {use_gpu} " \
        #                 #f"--min_length {min_length} " \
        #                 #f"--generate_motifs True " \

        attscore_path = os.path.join(save_dir, 'att_score.h5')
        pos_seq_path = os.path.join(save_dir, 'pos_seqs.txt')

        numbers = re.findall(r'\d+', experiment_key)
        if len(numbers) > 1:
            length_probe = numbers[-1]
        else:
            length_probe = numbers[0]
        if int(length_probe) == 20:
            window_size = 8
        else:
            window_size = 16

        logo_save_dir =  os.path.join(save_dir, 'motifs')
        
        ### running this to generate motifs
        motif_params =  f"python motif.py " \
                        f"--generate_motifs True " \
                        f"--generate_logo True " \
                        f"--attscore_path {attscore_path} " \
                        f"--pos_seq_path {pos_seq_path} " \
                        f"--save_dir {logo_save_dir} " \
                        f"--window_size {window_size} " \
                        f"--min_length {min_length} " \
                     

        status = os.system(motif_params)
        if status != 0:
            import sys 
            sys.exit()
        # break


# python motif.py --calc_score True \
# --n_sequences 1000 \
# --data_path ../seq_rAff_r6.txt \
# --model_dir /home/comp/csyuxu/aptdrug/ATF7_TGGGCG30NCGT \
# --save_dir /home/comp/csyuxu/aptdrug/test_folder \
# --vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
# --use_gpu True \
# --generate_motifs True \
# --min_length 4


# python motif.py --generate_motifs True \
# --save_dir /home/comp/csyuxu/aptdrug/test_folder \
# --attscore_path /home/comp/csyuxu/aptdrug/test_folder/att_score.h5 \
# --pos_seq_path /home/comp/csyuxu/aptdrug/test_folder/pos_seqs.txt \
# --window_size 16 \
# --min_length 4 \
# --generate_logo True
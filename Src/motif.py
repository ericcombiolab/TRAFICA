from transformers import BertConfig, BertTokenizer
from torch.utils.data import DataLoader, Dataset
# import torch.nn as nn
import torch
import numpy as np
import sys
import argparse
from models import Bert_seqClassification, Bert_seqRegression
import h5py


sys.path.append('..')
from utils import *



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def data_collate_fn(arr): 
    arr = np.array(arr)
    X = arr[:,0]
    y = arr[:,1]
    X = seq2kmer(list(X), k=4, min_length=10)
    collect_x = []
    for i in X:
        collect_x.append(' '.join(i))
    return collect_x, y

def contiguous_regions(condition, min_len=5):
    """
    Finds contiguous regions of True values in a boolean array. Returns a 2D
    array where the first column is the start index of the region and the
    second column is the end index.
    
    Arguments:
    condition -- boolean array 
    
    Keyword arguments:
    min_len -- int, specified minimum length threshold for contiguous region 
        (default 5)
    
    Returns:
    idx -- index of contiguous regions in sequence
    """
    # Find the indices where the condition changes
    diff = np.diff(condition)
    idx, = diff.nonzero()
    
    # Shift the start index by 1 if the condition starts with True
    if condition[0]:
        idx = np.r_[0, idx]
    
    # Shift the end index by 1 if the condition ends with True
    if condition[-1]:
        idx = np.r_[idx, condition.size-1]
    
    # Reshape the indices into two columns
    idx.shape = (-1, 2)
    
    # Filter out contiguous regions that are below the minimum length
    idx = idx[(idx[:,1] - idx[:,0]) >= min_len]
    
    return idx


def find_high_attention(score, min_len=5, **kwargs):
    """
    Finds contiguous high attention sub-regions in a score array. A sub-region
    is considered high attention if its score is greater than the mean score.
    
    Arguments:
    score -- numpy array of attention scores for a sequence
    
    Keyword arguments:
    min_len -- int, specified minimum length threshold for contiguous region 
        (default 5)
    **kwargs -- other input arguments:
        cond -- custom conditions to filter/select high attention 
            (list of boolean arrays)
    
    Returns:
    motif_regions -- indices of high attention regions in sequence
    """
    # Define the conditions for high attention regions
    cond1 = score > np.mean(score)
    
    # Combine the conditions with the logical AND operator
    #cond = np.logical_and(cond1, cond2)
    cond = cond1

    # Allow for custom conditions to be passed as kwargs
    if 'cond' in kwargs:
        cond = kwargs['cond']
        if isinstance(cond, list):
            cond = tuple(cond)
        # Combine custom conditions with the logical AND operator
        cond = np.logical_and(cond1, *cond)
    
    # Find the contiguous regions of high attention
    motif_regions = contiguous_regions(cond, min_len)
    
    motif_regions[:,1] += 3 # xu, 4-mer -> sequence
    return motif_regions

    
def merge_motifs(motif_seqs, min_len=5, align_all_ties=True, **kwargs):
    """
    Function to merge similar motifs in input motif_seqs.
    
    First sort keys of input motif_seqs based on length. For each query motif with length
    guaranteed to >= key motif, perform pairwise alignment between them.
    
    If can be aligned, find out best alignment among all combinations, then adjust start
    and end position of high attention region based on left/right offsets calculated by 
    alignment of the query and key motifs.
    
    If cannot be aligned with any existing key motifs, add to the new dict as new key motif.
    
    Returns a new dict containing merged motifs.
    
    Arguments:
    motif_seqs -- nested dict, with the following structure: 
        {motif: {seq_idx: idx, atten_region_pos: (start, end)}}
        where seq_idx indicates indices of pos_seqs containing a motif, and
        atten_region_pos indicates where the high attention region is located.
    
    Keyword arguments:
    min_len -- int, specified minimum length threshold for contiguous region 
        (default 5)
    
    align_all_ties -- bool, whether to keep all best alignments when ties encountered (default True)
    
    **kwargs -- other input arguments, may include:
        - cond: custom condition used to declare successful alignment.
            default is score > max of (min_len -1) and (1/2 times min length of two motifs aligned)
    
    Returns:
    merged_motif_seqs -- nested dict with same structure as `motif_seqs`
    """ 
    
    from Bio import Align
    
    ### TODO: modify algorithm to improve efficiency later
    aligner = Align.PairwiseAligner()
    aligner.internal_gap_score = -10000.0 # prohibit internal gaps
    
    merged_motif_seqs = {}
    for motif in sorted(motif_seqs, key=len): # query motif
        if not merged_motif_seqs: # if empty
            merged_motif_seqs[motif] = motif_seqs[motif] # add first one
        else: # not empty, then compare and see if can be merged
            # first create all alignment scores, to find out max
            alignments = []
            key_motifs = []
            for key_motif in merged_motif_seqs.keys(): # key motif
                if motif != key_motif: # do not attempt to align to self
                    # first is query, second is key within new dict
                    # first is guaranteed to be length >= second after sorting keys
                    alignment=aligner.align(motif, key_motif)[0] 
                    
                    # condition to declare successful alignment
                    cond = max((min_len -1), 0.5 * min(len(motif), len(key_motif))) 
                    
                    if 'cond' in kwargs:
                        cond = kwargs['cond'] # override
                        
                    if alignment.score >= cond: # exists key that can align
                        alignments.append(alignment)
                        key_motifs.append(key_motif)

            if alignments: # if aligned, find out alignment with maximum score and proceed
                best_score = max(alignments, key=lambda alignment: alignment.score)
                best_idx = [i for i, score in enumerate(alignments) if score == best_score]
                
                if align_all_ties:
                    for i in best_idx:
                        alignment = alignments[i]
                        key_motif = key_motifs[i]

                        # calculate offset to be added/subtracted from atten_region_pos
                        left_offset = alignment.aligned[0][0][0] - alignment.aligned[1][0][0] # always query - key
                        if (alignment.aligned[0][0][1] <= len(motif)) & \
                            (alignment.aligned[1][0][1] == len(key_motif)): # inside
                            right_offset = len(motif) - alignment.aligned[0][0][1]
                        elif (alignment.aligned[0][0][1] == len(motif)) & \
                            (alignment.aligned[1][0][1] < len(key_motif)): # left shift
                            right_offset = alignment.aligned[1][0][1] - len(key_motif)
                        elif (alignment.aligned[0][0][1] < len(motif)) & \
                            (alignment.aligned[1][0][1] == len(key_motif)): # right shift
                            right_offset = len(motif) - alignment.aligned[0][0][1]

                        # add seq_idx back to new merged dict
                        merged_motif_seqs[key_motif]['seq_idx'].extend(motif_seqs[motif]['seq_idx'])

                        # calculate new atten_region_pos after adding/subtracting offset 
                        new_atten_region_pos = [(pos[0]+left_offset, pos[1]-right_offset) \
                                                for pos in motif_seqs[motif]['atten_region_pos']]
                        merged_motif_seqs[key_motif]['atten_region_pos'].extend(new_atten_region_pos)

                else:
                    alignment = alignments[best_idx[0]]
                    key_motif = key_motifs[best_idx[0]]

                    # calculate offset to be added/subtracted from atten_region_pos
                    left_offset = alignment.aligned[0][0][0] - alignment.aligned[1][0][0] # always query - key
                    if (alignment.aligned[0][0][1] <= len(motif)) & \
                        (alignment.aligned[1][0][1] == len(key_motif)): # inside
                        right_offset = len(motif) - alignment.aligned[0][0][1]
                    elif (alignment.aligned[0][0][1] == len(motif)) & \
                        (alignment.aligned[1][0][1] < len(key_motif)): # left shift
                        right_offset = alignment.aligned[1][0][1] - len(key_motif)
                    elif (alignment.aligned[0][0][1] < len(motif)) & \
                        (alignment.aligned[1][0][1] == len(key_motif)): # right shift
                        right_offset = len(motif) - alignment.aligned[0][0][1]

                    # add seq_idx back to new merged dict
                    merged_motif_seqs[key_motif]['seq_idx'].extend(motif_seqs[motif]['seq_idx'])

                    # calculate new atten_region_pos after adding/subtracting offset 
                    new_atten_region_pos = [(pos[0]+left_offset, pos[1]-right_offset) \
                                            for pos in motif_seqs[motif]['atten_region_pos']]
                    merged_motif_seqs[key_motif]['atten_region_pos'].extend(new_atten_region_pos)

            else: # cannot align to anything, add to new dict as independent key
                merged_motif_seqs[motif] = motif_seqs[motif] # add new one
    
    return merged_motif_seqs


def make_window(motif_seqs, pos_seqs, window_size=24):
    """
    Function to extract fixed, equal length sequences centered at high-attention motif instance.
    
    Returns new dict containing seqs with fixed window_size.
    
    Arguments:
    motif_seqs -- nested dict, with the following structure: 
        {motif: {seq_idx: idx, atten_region_pos: (start, end)}}
        where seq_idx indicates indices of pos_seqs containing a motif, and
        atten_region_pos indicates where the high attention region is located.
    pos_seqs -- list, numpy array or pandas series of positive DNA sequences
    
    Keyword arguments:
    window_size -- int, specified window size to be final motif length
        (default 24)
    
    Returns:
    new_motif_seqs -- nested dict with same structure as `motif_seqs`s
    
    """ 
    new_motif_seqs = {}
    
    # extract fixed-length sequences based on window_size
    for motif, instances in motif_seqs.items():
        new_motif_seqs[motif] = {'seq_idx':[], 'atten_region_pos':[], 'seqs': []}
        for i, coord in enumerate(instances['atten_region_pos']):
            atten_len = coord[1] - coord[0]
            if (window_size - atten_len) % 2 == 0: # even
                offset = (window_size - atten_len) / 2
                new_coord = (int(coord[0] - offset), int(coord[1] + offset))
                if (new_coord[0] >=0) & (new_coord[1] < len(pos_seqs[instances['seq_idx'][i]])):
                    # append
                    new_motif_seqs[motif]['seq_idx'].append(instances['seq_idx'][i])
                    new_motif_seqs[motif]['atten_region_pos'].append((new_coord[0], new_coord[1]))
                    new_motif_seqs[motif]['seqs'].append(pos_seqs[instances['seq_idx'][i]][new_coord[0]:new_coord[1]])
            else: # odd
                offset1 = (window_size - atten_len) // 2
                offset2 = (window_size - atten_len) // 2 + 1
                new_coord = (int(coord[0] - offset1), int(coord[1] + offset2))
                if (new_coord[0] >=0) & (new_coord[1] < len(pos_seqs[instances['seq_idx'][i]])):
                    # append
                    new_motif_seqs[motif]['seq_idx'].append(instances['seq_idx'][i])
                    new_motif_seqs[motif]['atten_region_pos'].append((new_coord[0], new_coord[1]))
                    new_motif_seqs[motif]['seqs'].append(pos_seqs[instances['seq_idx'][i]][new_coord[0]:new_coord[1]])

    return new_motif_seqs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--calc_score', required=False, type=bool, default=False, help="Calc attention scores for input sequences")    
    parser.add_argument('--n_sequences', required=False, type=int, default=1000, help="Num of sequence used to derive sequence motifs; if None, using all sequences of the input data")    
    
    parser.add_argument('--use_gpu', default=False, type=bool, help="Turn on to use gpu to train model; False: using cpu, True: using gpu if available.")
    parser.add_argument('--save_dir', required=True,type=str, help="The saving path of the trained model")
    parser.add_argument('--data_path', required=False, type=str, help="The path of the data folder")
    parser.add_argument('--vocab_path', required=False, type=str, help="The path of the vocabulary")
    parser.add_argument('--model_dir', required=False, type=str, help=".")    

    parser.add_argument('--generate_motifs', required=False, type=bool, default=False, help=".")  
    parser.add_argument('--pos_seq_path', required=False, type=str, help=".")
    parser.add_argument('--min_length', required=False, default=4, type=int, help="The minimum length of the contigous high-attention tokens.")
    parser.add_argument('--window_size', required=False, default=16, type=int, help="The window size for scanning the region around the motif in a sequence")

    parser.add_argument('--attscore_path', required=False,  type=str, help=".")

    parser.add_argument('--generate_logo', required=False, type=bool, default=False, help=".")  
 
    args = parser.parse_args()



    if args.calc_score:
        set_seeds(3047)  # torch.manual_seed(3407) is all you need

        # test device
        if args.use_gpu:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = 'cpu'

        # saving
        create_directory(args.save_dir)

        # load pre-trained model & tokenizer initialization
        Tokenizer = BertTokenizer(args.vocab_path, do_lower_case=False, model_max_length=512)

        # datasets of each experiment
        # data = load_datasets_test(args.data_path, batch_size=32, k=4)
        data = pd.read_table(args.data_path).values
        if args.n_sequences:
            data = data[:args.n_sequences] 
        pos_seqs =  data[:, 0]

        dataset = MyDataset(data)
        dataloader = DataLoader(dataset, 32, shuffle=False, collate_fn=data_collate_fn)
        
        configuration = BertConfig.from_pretrained(args.model_dir)    
        model = Bert_seqRegression(configuration, args.model_dir, pool_strategy='mean', fine_tuned=True, out_attention=True) 
        if torch.cuda.is_available():
            model.to(device)
        model.eval()

        att_score_collect = []
        label_collect = []
        kmer_collect = []
        count = 0
        for batch in dataloader:
            batch_data_kmer = batch[0]       
            batch_label = batch[1]

            inputs = Tokenizer(batch_data_kmer, return_tensors="pt", padding=True)
            pred, attentions = model(inputs.to(device))
            att = torch.stack(attentions)
            att = torch.permute(att, (1, 0, 2, 3, 4)) # batch, n_layer, n_heads, len_seq, len_seq
            att = att[:,:,:,1:-1,1:-1] # discard [CLS] and [SEP]
            score = torch.sum(att,dim=3) # attention from other tokens
            score = torch.mean(score,dim=2) # average attention from different heads
            score = torch.mean(score,dim=1) # average attention from different layers

            att_score_collect.append(score.cpu().detach().numpy())
            label_collect += batch_label.tolist()
            kmer_collect += (batch_data_kmer)

            # if count >1:
            #     break
            # count+=1

        att_score = np.concatenate(att_score_collect, axis=0)
    
        with h5py.File(os.path.join(args.save_dir,'att_score.h5'), 'w') as f:
            f.create_dataset('score', data=att_score)

        save_txt_single_column(os.path.join(args.save_dir, 'pos_seqs.txt'), pos_seqs.tolist())


    if args.generate_motifs:
        if not args.calc_score:
                
            with h5py.File(args.attscore_path, 'r') as f:
                att_score = f['score'][:]

            #data = pd.read_table(args.data_path).values
            pos_seqs = np.array( load_txt_single_column(args.pos_seq_path) )

            create_directory(args.save_dir)

        
        motif_seqs = {}
        k = 4 
        m_len = args.min_length
        for i, score in enumerate(att_score):
        #i = 2
            motif_regions = find_high_attention(att_score[i], min_len=m_len) # min_len is important to dicovery the short motifs
            for motif_idx in motif_regions:
                seq = pos_seqs[i][motif_idx[0]:motif_idx[1]]
                if seq not in motif_seqs:
                    motif_seqs[seq] = {'seq_idx': [i], 'atten_region_pos':[(motif_idx[0],motif_idx[1])]}
                else:
                    motif_seqs[seq]['seq_idx'].append(i)
                    motif_seqs[seq]['atten_region_pos'].append((motif_idx[0],motif_idx[1]))


        merged_motif_seqs = merge_motifs(motif_seqs, min_len=m_len,  align_all_ties = True)   
        merged_motif_seqs = make_window(merged_motif_seqs, pos_seqs, window_size=args.window_size)

        motif_seq_folder = os.path.join(args.save_dir, 'motif_seq')
        create_directory(motif_seq_folder)
        for motif, instances in merged_motif_seqs.items():
            # saving to files
            with open( os.path.join(motif_seq_folder, 'motif_{}_{}.txt'.format(motif, len(instances['seq_idx'])) ), 'w') as f:
                for seq in instances['seqs']:
                    f.write(seq+'\n')


        if args.generate_logo:
            logo_out_dir = os.path.join(args.save_dir, 'motif_logo')


            create_directory(logo_out_dir)

            seq_files = os.listdir(motif_seq_folder)
            for file in seq_files:
                filename = file.split('.')[0]
                out_format = 'png_print'
                weblogo_params = f"weblogo " \
                                f"-f {os.path.join(motif_seq_folder, file)} " \
                                f"-o {os.path.join(logo_out_dir, filename+'.png')} " \
                                f"-F {out_format} " \
                                f"--number-interval {1} " \
                                f"--number-fontsize {4} " \
                                f"--errorbars no " \
                                f"--fineprint ' ' " \
                                f"--color-scheme classic "
                status = os.system(weblogo_params)
                if status != 0:
                    import sys
                    sys.exit()
 
  

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

# weblogo -f ../motif/MA0470.2.transfac -D transfac -o ../motif/MA04702.png -F png_print --number-interval 1 --number-fontsize 4 --errorbars no --color-scheme classic -s large --scale-width no --fineprint ' ' --stack-width 20
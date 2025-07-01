import os
import itertools
import json
import numpy as np
from typing import List
import random
from typing import List#, Union, Dict
# from collections import defaultdict
import sentencepiece as spm
import h5py
import pandas as pd 
import torch 


def save_txt_single_column(data, save_dir='./', filename='save_test.txt'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with open(os.path.join(save_dir, filename),'w') as f:
        for i in data:
            f.write(str(i)+'\n')
            

def DNA_VocabularyGeneration(mode, save_path, data_path=None, BPE_folder=None):
    special_tokens = ['[PAD]', '[UNK]','[CLS]','[SEP]','[MASK]']
    bases = ['A', 'T', 'C', 'G']
    if mode == 'Base-level':
        vocab = special_tokens + bases
    elif mode == '3_mer':
        vocab = special_tokens + [''.join(p) for p in itertools.product(bases, repeat=3)]
    elif mode == '4_mer':
        vocab = special_tokens + [''.join(p) for p in itertools.product(bases, repeat=4)]
    elif mode == '5_mer':
        vocab = special_tokens + [''.join(p) for p in itertools.product(bases, repeat=5)]
    elif mode == '6_mer':
        vocab = special_tokens + [''.join(p) for p in itertools.product(bases, repeat=6)]
    elif mode == 'BPE':
        # BPE_train(BPE_folder, data_path)
        df = pd.read_table(os.path.join(BPE_folder, 'dna_BPE_model.vocab'), header=None)
        BPE_tokens = df.iloc[:,0].values.ravel()
        vocab = special_tokens +  BPE_tokens[1:].tolist()


    f =open(save_path,'w')
    for i in vocab:
        f.write(i+'\n')
    f.close()


def BPE_train(BPE_folder, data_path):
    '''
    sentencepiece document: https://github.com/google/sentencepiece/blob/master/doc/options.md
    '''
    if not os.path.exists(BPE_folder):
        os.makedirs(BPE_folder)
        
    # extract DNA sequences -> corpus
    hdf5_file = h5py.File(data_path, 'r')
    sequences = hdf5_file['sequence'][:]
    total_num = len(sequences)
    count = 0
    f = open(os.path.join(BPE_folder, 'dna_sequences.txt'),'w')
    for seq in sequences:
        count+=1
        if (count %1000) ==0:
            print(f"processing:\t{count}/{total_num}")
        f.write(seq.decode('utf-8')+'\n')
    f.close()
    
    # train BPE model
    input_file = os.path.join(BPE_folder, 'dna_sequences.txt')
    vocab_size = 4097 # 4096 + 1:  add 1 to include <unk> which can no be disable using sentencepiece tool  
    model_prefix = os.path.join(BPE_folder, 'dna_BPE_model')
    spm.SentencePieceTrainer.train(input=input_file, 
                                model_prefix=model_prefix, 
                                vocab_size=vocab_size, 
                                model_type='BPE',
                                add_dummy_prefix=False,    
                                bos_id=-1,  
                                eos_id=-1, 
                                ) 
    
def save_json_file(save_path, dict_data):
    with open(save_path, "w") as write_file:
            json.dump(dict_data, write_file, indent=4)
            

def create_directory(path):
    """
    Creates a directory at the specified path if it doesn't already exist.
    
    Args:
    path (str): The directory path to create.
    """
    if not os.path.exists(path): # Check if the directory already exists
        os.makedirs(path) # Create the directory and any necessary parent directories


def batch_kmerize(sequences, k=3):

    valid_bases = {'A', 'T', 'C', 'G'}
    results = []

    for seq_idx, seq in enumerate(sequences):
        seq = seq.upper()
  
        filtered_seq = [b for b in seq if b in valid_bases]
        seq_len = len(filtered_seq)
        
        if seq_len < k:
            continue
            
        kmers = np.empty(seq_len - k + 1, dtype=object)
        
        for i in range(seq_len - k + 1):
            kmer = ''.join(filtered_seq[i:i+k])
            kmers[i] = kmer
    
        results.append(kmers.tolist())

    return results



def piece_sequences(
    sequences: List[str],
    tokenization_mode: str = 'Base-level',
    sp=None) -> List[str]:


    if tokenization_mode == 'Base-level':
        # (shape: [n_seq, max_len])
        max_len = max(len(seq) for seq in sequences)                                # max number of base tokens 
        char_matrix = np.array([list(seq.ljust(max_len)) for seq in sequences])
        
        # Base-level
        space_mask = np.zeros((char_matrix.shape[0], char_matrix.shape[1] * 2 - 1), dtype='U1')
        space_mask[:, ::2] = char_matrix
        space_mask[:, 1::2] = ' '
        return [''.join(row).strip() for row in space_mask]
    
    elif tokenization_mode == 'BPE':
        BPE_tokens = sp.encode_as_pieces(sequences)
        return [' '.join(tokens) for tokens in BPE_tokens]
    
    elif tokenization_mode in ['3_mer','4_mer','5_mer','6_mer']:
        # obtain k
        k = int(tokenization_mode[0])
        kmer_tokens = batch_kmerize(sequences, k)
        
        return [' '.join(tokens) for tokens in kmer_tokens]
    
    
    else:
        raise ValueError(f"unsupported okenization_mode: {tokenization_mode}")
    
    
    
def slice_sequences(sequences: List[str], 
                   low: float = 0.1, 
                   high: float = 1.0,
                   ensure_full_prob: float = 0.1) -> List[str]:
    """
    对DNA序列随机切片,确保有一定概率保留完整序列 ratio=1.0
    
    Args:
        sequences: 
        low:  (.1)
        high:  (1.0)
        ensure_full_prob:  10%
    
    Returns:
        
    """
    n = len(sequences)
    
    ratios = np.random.uniform(low=low, high=high, size=n)
    
    if ensure_full_prob > 0:
        full_mask = np.random.random(n) < ensure_full_prob 
        ratios[full_mask] = 1.0  
    
    sliced_seqs = []
    for seq, ratio in zip(sequences, ratios):
        seq_len = len(seq)
        sub_len = int(seq_len * ratio)
        
        if ratio == 1.0:  
            sliced_seqs.append(seq)
        else:
            start = np.random.randint(0, seq_len - sub_len + 1)
            end = start + sub_len
            sliced_seqs.append(seq[start:end])
    
    return sliced_seqs




def mask_dna_sequence(
    tokens_list: List[str],
    mask_token: str = "[MASK]",
    mask_ratio: float = 0.15,
    continuous_tokens: int = 1) -> List[str]:
    """
    
    Args:
        tokens_list:  ( ["A C G T", "T T A G"])
        mask_token:  (default [MASK])
        mask_ratio:  (0-1)
        continuous_tokens: 
    
    Returns:
        
    """
    # 
    token_arrays = [seq.split() for seq in tokens_list]
    max_len = max(len(arr) for arr in token_arrays)
    padded_arrays = [arr + [''] * (max_len - len(arr)) for arr in token_arrays]
    token_matrix = np.array(padded_arrays, dtype=object)  # shape: (n_sequences, max_len)
    
    # batch operation
    n_sequences, seq_len = token_matrix.shape
    # total_mask_tokens = max(1, round(seq_len * mask_ratio))
    
    # init mask matrix
    mask_matrix = np.zeros((n_sequences, seq_len), dtype=bool)

    for i in range(n_sequences):
        
        actual_len = len(token_arrays[i])
        if actual_len == 0:
            continue

        current_mask_tokens = max(1, round(actual_len * mask_ratio))
        
        if continuous_tokens > 1:
            num_blocks = min(current_mask_tokens // continuous_tokens, actual_len // continuous_tokens)
            starts = random.sample(range(actual_len - continuous_tokens + 1), num_blocks)
            for start in starts:
                mask_matrix[i, start:start+continuous_tokens] = True
            remaining = current_mask_tokens - num_blocks * continuous_tokens
        else:
            remaining = current_mask_tokens
        
        
        if remaining > 0:
            available = [j for j in range(actual_len) if not mask_matrix[i, j]]
            if available:
                for pos in random.sample(available, min(remaining, len(available))):
                    mask_matrix[i, pos] = True
    
    # 
    token_matrix[mask_matrix] = mask_token
    
    # return list of strings
    masked_sequences = [
        ' '.join(filter(None, row))  
        for row in token_matrix
    ]
    
    return masked_sequences

def set_seeds(seed_val): 
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


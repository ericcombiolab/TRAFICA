import pandas as pd
import numpy as np
import os
import sys
import torch
import random
import six
import argparse
import json
from torch.utils.data import Dataset
import re 

# # token
# vocab = ['A', 'T', 'C', 'G', 'N', 'Mask', 'CLS', 'PAD', 'SEP'] # default vocabulary for char-token strategy
# vocab_special = ['Mask', 'CLS', 'PAD', 'SEP', 'UNK']



##################### customary behavior #####################
## breakpoint debug 
def debug_breakpoint(context=None):
    if context!=None:
        if type(context) == list:
            for c in context:
                print(c)
        else:
            print(context)
    sys.exit()


## avoid cmd error when a file name includes ()
def escape_special_chars(s):
    pattern = r'([\(\)])'
    repl = r'\\\1'
    return re.sub(pattern, repl, s)


## iterate all files in a folder including the subfolders
def iter_files_in_all_subfolders(folder):
    ## for each folder
    def _single_folder(folder):
        buffer = []
        files = os.listdir(folder) # nodes
        if type(files) == list:
            for file in files:
                buffer.append(os.path.join(folder, file))
        else:
            buffer.append(os.path.join(folder, files))                  
        return buffer
        
    ## processing
    buf = []
    if type(folder) == list:
        for f in folder:
            if os.path.isdir(f):
                buf += _single_folder(f) # still is folder, getting the files it contains
            else:
                buf += [f] # reach leaf (not folder)              
    else:
        if os.path.isdir(folder):
            buf += _single_folder(folder) # still is folder, getting the files it contains
        else:
            buf += [folder] # reach leaf (not folder)
    
    ## iteration condition
    for i in buf: 
        if os.path.isdir(i): # contains folders
            return iter_files_in_all_subfolders(buf) 
    return buf  # stop iteration
       

## logger for some running information
def print_to_log(path='./', save_name='log.txt'):
    class logger(object):
        def __init__(self, filename='log.txt', path = './'):
       
            self.terminal = sys.stdout
            self.log = open(os.path.join(path,filename),'a',encoding='utf8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    sys.stdout = logger(path=path, filename=save_name)

## Get dimension of python list object
def get_list_dim(list_, n_dims=0):
    if type(list_) == list:
        n_dims += 1
        return get_list_dim(list_[0], n_dims)
    else:
        return n_dims

## saving auroc and auprc dynamically into two json files
def saving_test_performance(save_folder, result, file_name, exp):
    if os.path.exists(os.path.join(save_folder, file_name + ".json")):
        with open(os.path.join(save_folder, file_name + ".json"), "r", encoding='UTF-8') as read_file:
            load_test = json.load(read_file)

        load_test[exp] = result  

        with open(os.path.join(save_folder, file_name+".json"), "w") as write_file:
            json.dump(load_test, write_file, indent=4)
    else:
        with open(os.path.join(save_folder, file_name+".json"), "w") as write_file:
            json.dump({exp:result}, write_file, indent=4)


def remove_noneElements_in_list(data):
    return [i for i in data if i!=""]



def equal_width_bins(array, num_bins):
    """
    Divide the array into equal-width bins.

    Args:
        array (numpy.ndarray): Input array.
        num_bins (int): Number of bins to create.
        
    Returns:
        numpy.ndarray: Array containing the bin indices for each value in the input array.
    """
    # Calculate the width of each bin
    array = array[:,1] # affinity column

    min_value = np.min(array)
    max_value = np.max(array)
    bin_width = (max_value - min_value) / num_bins
    
    # Use numpy.histogram function to compute the frequencies of each bin
    hist, _ = np.histogram(array, bins=num_bins, range=(min_value, max_value))
    
    # Use numpy.digitize function to map the values in the array to their corresponding bins
    bin_indices = np.digitize(array, bins=np.arange(min_value, max_value, bin_width))
    
    return bin_indices


def quantile_bins(array, num_bins):
    """
    Divide the array into quantile-based bins.

    Args:
        array (numpy.ndarray): Input array.
        num_bins (int): Number of bins to create.
        
    Returns:
        numpy.ndarray: Array containing the bin indices for each value in the input array.
    """
    array = array[:,1] # affinity column

    bin_indices = np.zeros_like(array, dtype=int)
    percentiles = np.linspace(0, 100, num_bins+1)
    bin_edges = np.percentile(array, percentiles)
    
    for i in range(num_bins):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i+1]
        bin_indices[(array >= lower_bound) & (array <= upper_bound)] = i
    
    return bin_indices


def sample_from_bins(array, bin_indices):
    """
    Sample one value from each bin and combine them into a new array.

    Args:
        array (numpy.ndarray): Input array.
        bin_indices (numpy.ndarray): Array containing the bin indices for each value in the input array.
        
    Returns:
        numpy.ndarray: Array containing one sampled value from each bin.
    """
    unique_bins = np.unique(bin_indices)  # Get the unique bin indices

    sampled_values = []
    for bin_index in unique_bins:
        bin_values = array[bin_indices == bin_index]  # Get the values in the current bin

        # Sample one value from the bin
        random_indices = np.random.choice(len(bin_values), 1, replace=False)[0]
        sampled_value = bin_values[random_indices]
      
        sampled_values.append(sampled_value)
 
    sampled_array = np.array(sampled_values)
    return sampled_array

##################### file operation #####################
def Dir_check_and_create(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        return dir_path

def list_name_inDir(dir_path):
    result = [filename for filename in os.listdir(dir_path)]
    return result

def save_txt_single_column(save_path, data):
    f = open(save_path,'w')
    if type(data) == list:
        for i in data:
            f.write(str(i)+'\n')
    else:
        f.write(str(data)+'\n')
    f.close()


def load_txt_single_column(save_path):
    f = open(save_path,'r')
    collect = []
    for line in f:
        collect.append(line.strip())
    f.close()
    return collect    
    


def split_one_seq_one_file(data_dir, save_dir, level_sub_folder=1, split_chunks=None, chunk_start=0):   # split_chunks: num of files in a chunk; this is to avoid the consumption of disk space when processing large-scale datasets (experiments)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    progress_count = 0

    if level_sub_folder>1: 
        kmer_files = iter_files_in_all_subfolders(data_dir)
        ## disk space limitation: split into subsets to run many times for running all experiments
        if split_chunks:
            if len(kmer_files) < (chunk_start+split_chunks):
                kmer_files = kmer_files[chunk_start : ]
            else:
                kmer_files = kmer_files[chunk_start : int(chunk_start+split_chunks)]
        
        for i in kmer_files:
            k_mers = load_data_kmer_format(i) # load data

            ## subfolders for each .kmer file 
            sub_folder_collect = []
            for level in range(1, level_sub_folder+1):
                sub_folder_collect.append(i.split('/')[-level])
            sub_folder_collect.reverse()

            new_dir_ = save_dir
            for z in range(len(sub_folder_collect)): 
                new_dir_ = os.path.join(new_dir_, sub_folder_collect[z])  
                if not os.path.exists(new_dir_):
                    os.mkdir(new_dir_)


            ## split into line files
            for j in range(len(k_mers)): # each sequence/kmer (row)
                
                f = open(os.path.join(new_dir_, str(j)), 'w')
                for k in k_mers[j]:
                    f.write(k+'\t')
                f.close()

                progress_count+=1
                if progress_count%100000 == 0:
                    print('finished:\t', progress_count)
            

    else:
        # for pretrain data (compress all experiments data into one folder)
        files = os.listdir(data_dir)
        progress_count = 0
        for i in files:
            k_mers = load_data_kmer_format(os.path.join(data_dir, i))
            for j in range(len(k_mers)):

                f = open(os.path.join(save_dir, i+'.'+str(j)), 'w')
                for k in k_mers[j]:
                    f.write(k+'\t')
                f.close()

                progress_count+=1
                if progress_count%100000 == 0:
                    print('finished:\t', progress_count)


def create_directory(path):
    """
    Creates a directory at the specified path if it doesn't already exist.
    
    Args:
    path (str): The directory path to create.
    """
    if not os.path.exists(path): # Check if the directory already exists
        os.makedirs(path) # Create the directory and any necessary parent directories


def get_file_extension(file_path):
    """
    Returns the file extension of the given file path.
    
    Args:
        file_path (str): The absolute path of the file.
        
    Returns:
        str: The file extension of the file.
    """
    # Use splitext function to get the file name and extension
    _, extension = os.path.splitext(file_path)
    # Remove the dot from the extension
    extension = extension[1:]
    return extension


##################### k-mer vocabularies construction #####################
## Enumerate all possible k-mers
def expand_one(seq_list, k, base_type='DNA', add_unknown_base=False):
    if base_type == 'DNA':
        expand_chars = ['A', 'T', 'C', 'G']  
    elif base_type == 'RNA': 
        expand_chars = ['A', 'U', 'C', 'G']    
    else:
        print('Base type error when expanding k-mer tree!') 
        return None
 
    if add_unknown_base==True:
        expand_chars+=['N']

    # recursive function  
    if len(seq_list[0]) < k:    
        buffer = []
        for seq in seq_list:   # expand n-th position for each sequence ( len(seq)==n-1 )              
            _buffer = []
            for j in expand_chars:
                _buffer.append(seq + j)
            buffer += _buffer         
        return expand_one(buffer, k, base_type, add_unknown_base) # combat 'return value == None'
    else:
        return seq_list
        
# Users can start with different bases composition
def expand_from_specifyString(k, base_string):   
    possible_strings = expand_one(base_string, k) # possible_k_mers
    return possible_strings

# k-mer vocabulary construction
def create_k_mer_vocab(k, base_type='DNA', add_unknown_base=False):
    if base_type == 'DNA':
        bases = ['A', 'T', 'C', 'G']#, 'N']   
    elif base_type == 'RNA': 
        bases = ['A', 'U', 'C', 'G', 'N']    
    else:
        print('Base type error when creating k-mer vocabulary!') 
        return None

    if add_unknown_base==True:
        bases+=['N'] 
    vocab = expand_one(bases, k, base_type=base_type, add_unknown_base=add_unknown_base) # possible_k_mers
    return vocab, len(vocab)


##################### sequence & k-mer #####################
## Convert sequence(s) to k-mer(s)
def seq2kmer(sequences, k, min_length=10):
    def _seq2kmer_for_one_seq(seq, k):
        n_base = len(seq)
        if n_base < k:
            print('sequence length error:\t', seq, str(n_base))
            import sys
            sys.exit()

        start = 0
        collect = []
        for i in range( int( n_base - (k-1) ) ):
            _mer = seq[ start : start+k ]
            
            if type(_mer) == list:  # transform list chars into a string
                _mer = ''.join(_mer)
            
            collect.append(_mer)
            start += 1
        return collect

    if get_list_dim(sequences) > 0:     # if input are multiple sequences
        collect_for_each_seq = []
        for seq in sequences:
            if len(seq) >= min_length:             
                collect_for_each_seq.append(_seq2kmer_for_one_seq(seq, k))
        return collect_for_each_seq
    else: 
        if len(sequences) < min_length:                                      # if input is single sequence
            print('sequence length less than the minimum:\t',len(sequences), min_length)
            pass
        else:
            return _seq2kmer_for_one_seq(sequences, k)


## Convert sequence(s) to k-mer(s) with the special tokens 
# discard it if using huggingface bert tokenizer
def seq2kmer_with_special_symbols(sequences, k):
    k_mer_from_seqs = seq2kmer(sequences, k)
    if get_list_dim(k_mer_from_seqs) == 1:  # single sequence
        k_mer_from_seqs_with_SS = ['CLS'] + k_mer_from_seqs + ['SEP'] 
        return k_mer_from_seqs_with_SS
    k_mer_from_seqs_with_SS = [['CLS'] + i + ['SEP'] for i in k_mer_from_seqs]
    return k_mer_from_seqs_with_SS          # multiple sequences


## Batch sequences files operation
# results will be stored into the new folder named 'XX_k_mer'
def pre_process_seq2kmer_files(data_dir, k=4, special_symbols=False, label=False, level_sub_folder=1, min_length=20):
    # for large-scale data
    seq_files = iter_files_in_all_subfolders(data_dir)
    count = 0
    new_dir = data_dir + '_' + str(k) + 'mer'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    for i in seq_files:
        j = i.split('/')[-1]
        name, _ = j.split('.')
        new_name = name + '.' + str(k) + 'mer'

        ## loading sequences (with labels)
        if label:
            seqs_labels = pd.read_table(i, header=None, delim_whitespace=True).values
            seqs = seqs_labels[:, 0].tolist()
            labels = seqs_labels[:, 1].tolist()          
        else:
            seqs = pd.read_table(i, header=None, delim_whitespace=True).values.ravel().tolist()

        ## optional
        if special_symbols:
            k_mers = seq2kmer_with_special_symbols(seqs, k)
        else:
            k_mers = seq2kmer(seqs, k, min_length=min_length)

        ## for file tree, creat saving folders(subfolders)
        if level_sub_folder > 1:
            sub_folder_collect = []
            for level in range(2, level_sub_folder+1):
                sub_folder_collect.append(i.split('/')[-level])
            sub_folder_collect.reverse()

            new_dir_ = new_dir
            for z in range(len(sub_folder_collect)): 
                new_dir_ = os.path.join(new_dir_, sub_folder_collect[z])  
                if not os.path.exists(new_dir_):
                    os.mkdir(new_dir_)
        else:
            new_dir_ = new_dir

        ## saving processed k-mers
        if label:
            f = open(os.path.join(new_dir_, new_name), 'w')
            for i in range(len(k_mers)):
                for kmer in k_mers[i]:
                    f.write(kmer + '\t')
                f.write(str(labels[i]))
                f.write('\n')
            f.close()
        else:
            f = open(os.path.join(new_dir_, new_name), 'w')
            for row in k_mers:
                for kmer in row:
                    f.write(kmer + '\t')
                f.write('\n')
            f.close()

        count+=1
        print('the number of processed k-mer files :\t%d/%d' % (count, len(seq_files)))


# def get_aminoacid_type_from_seqfile(file_path):
#     seqs = pd.read_table(file_path, header=None).values.ravel()
#     collect = []
#     for seq in seqs:
#         collect += list(set([char for char in seq]))
#         collect = list(set(collect))
#     return collect, len(collect)

##################### Bert source code: creat multi-head attention matrices  #####################
# these functions were modified from google research bert source code: 'modeling.py' (yuxu)
def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.
    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types): # compatible python 2 and 3; six.integer_types: 'long' and 'int' in python2 while only 'int' in python3;
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = len(tensor.shape) # tensor dimensions

    if actual_rank not in expected_rank_dict:
        scope_name = tensor.names
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
        name: Optional name of the tensor for the error message.
    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
  """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = list(tensor.shape)

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tensor.shape
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    #from_shape=[batch_size,from_seq_length]
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    #len(tensor.shape) == 2 or 3 only
    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    # to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
    to_mask = torch.reshape(to_mask, [batch_size, 1, to_seq_length]).float() # for pytorch

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = torch.ones([batch_size, from_seq_length, 1]).float()

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask    # shape = [batch_size, from_seq_length, to_seq_length]

    return mask



##################### Manually Padding & Token<->Index #####################
def seq_padding(seq, MAX_LEN = 512):   # add 'PAD' symbol
    if get_list_dim(seq) > 1: # multiple sequences (list)
        collect = []
        for i in seq:
            n_pad = MAX_LEN - len(i)
            if n_pad > 0:
                collect.append(i + ['PAD' for j in range(int(n_pad))])
        return collect
    elif len(np.array(seq, dtype=object).shape) == 1:  # a sequence
        return seq + ['[PAD]' for j in range(int(MAX_LEN - len(seq)))]
    else:
        print('sequence padding error, check the shape of input sequence.')
        import sys
        sys.exit()


def convert_char_2_num(seq, vocab_, return_tensor=True):
    def _single_sequence(seq):
        collect = []
        for i in seq:
            if i in vocab_:
                collect.append(vocab_.index(i))
            else:
                collect.append(vocab_.index('[UNK]')) # skip some token error caused by char 
                # print('error in char:\t', i)
                # sys.exit()         
        return collect

    if get_list_dim(seq) > 1:
        collect = []
        for s in seq:
            collect.append(_single_sequence(s))
        
        if return_tensor:
            return torch.tensor(collect).long()
        else:
            return collect
    else:
        if return_tensor:
            return torch.tensor(_single_sequence(seq)).long()
        else:
            return _single_sequence(seq)
   

def convert_num_2_char(idx, vocab):   
    char_matrix = np.array(vocab)[idx]  # with [CLS] and [PAD]
    if len(char_matrix.shape) > 1:
        char_matrix = char_matrix[:,1:] # remove [CLS]
        # remove [PAD] and convert into sequences (strings) 
        collect = []
        for row in char_matrix:
            if '[PAD]' in row: # avoid 'PAD' no existing in predicted results
                idx_pad = np.argwhere(row == '[PAD]').ravel()
                seq = row[ : idx_pad[0] ]    # idx of first [PAD]
            else:
                seq = row
            collect.append(''.join(seq))
        return collect
    else:
        row = char_matrix[1:]
        if '[PAD]' in row:
            idx_pad = np.argwhere(row == '[PAD]').ravel()
            seq = row[ : idx_pad[0] ]
        else:
            seq = row   
        return ''.join(seq)

# ATCG -> one-hot encoding
def encode_seq2onehot(seqs, types='DNA', return_tensor=False):
    if types == 'DNA':
        char2num = {'A':0,'T':1,'C':2, 'G':3}
    elif types == 'RNA':
        char2num = {'A':0,'U':1,'C':2, 'G':3}

    def _single_sequence(seq, char2num, _return_tensor=False):
        onehot_matrix = np.zeros((len(char2num.keys()), len(seq))) 
        for i in range(len(seq)):
            onehot_matrix[char2num[seq[i]],i] = 1
        
        if _return_tensor:
            return torch.tensor(onehot_matrix)
        return onehot_matrix
    
    if type(seqs) == list: 
        collect=[]
        for seq in seqs:
            collect.append(_single_sequence(seq, char2num, _return_tensor=return_tensor))
        
        if return_tensor:
            matrix = torch.stack(collect)
            return matrix
        else:
            return collect
    else:
        single_matrix = _single_sequence(seqs, char2num, _return_tensor=return_tensor)
        return single_matrix.unsqueeze(0)

##################### Learning rate warmup strategy #####################
class WarmupLR:
    def __init__(self, optimizer, max_lr, num_warm) -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        #self.lr = [group['lr'] for group in self.optimizer.param_groups]
        self.lr = max_lr
        self.num_step = 0

    def __compute(self, lr) -> float:
        return 10 * lr * min(self.num_step ** (-0.5), self.num_step * self.num_warm ** (-1.5))

    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr] 
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]

    def get_lr(self):
        return self.lr


class WarmupLR_Linear:
    def __init__(self, optimizer, max_lr, num_warm, num_allsteps) -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        self.lr = max_lr
        self.num_step = 0
        self.num_allsteps = num_allsteps

    def __compute(self, lr) -> float:
        if self.num_step <= self.num_warm:
            return lr * ( self.num_step/self.num_warm )
        else:   # linear decay
            
            return lr * (1- ( (self.num_step-self.num_warm) / (self.num_allsteps-self.num_warm) ) )

    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr] 
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]

    def get_lr(self):
        return self.lr



##################### MASK operation #####################
def mask_strategy_contiguous(input, vocab, ratio=0.15): 
    """
    The strategy used in DNABert
    """
    len_batch_seqs = torch.argwhere(input == vocab.index('[SEP]'))[:,1] - 1
    n_words_mask = len_batch_seqs * ratio
    n_words_mask = n_words_mask.round().int()

    rand_value = torch.randint(1, int((1-ratio)*100), [input.shape[0]]) / 100 # mask start pointer
    pos_start_mask = len_batch_seqs * rand_value
    pos_start_mask = pos_start_mask.int()
    
    rand_mask = torch.zeros(input.shape)
    for i in range(len(rand_mask)):
        rand_mask[i, pos_start_mask[i]: pos_start_mask[i] + n_words_mask[i]] = 1
    rand_mask = rand_mask.bool()
    return rand_mask


def mask_strategy_contiguous_split(input, vocab, ratio=0.15, n_phrase_mask=1): 
    """
    Modified version of contiguous k-mers mask strategy;
    n_phrase_mask should be smaller than the length of sequences (in k-mers format)
    """
    def get_random_mask_phrases_idx(num_phrases, n_phrase_mask, phrase_size):
        idx_collect=[]
        for i in range(len(num_phrases)):
            idx = random.sample([(int( j*phrase_size[i] ) + 1) for j in range(num_phrases[i])], n_phrase_mask) # add 1 for skipping [CLS]
            idx_collect.append(idx)
        return idx_collect

    len_batch_seqs = torch.argwhere(input == vocab.index('[SEP]'))[:,1] - 1 # length of each sequence (in k-mer format)
    phrase_size = len_batch_seqs / n_phrase_mask * ratio  # the num of words to be masked (for a phrase) 
    phrase_size = phrase_size.round().int() # rounding the numbers
    phrase_size[ phrase_size==0 ] == 1 # avoid denominator to be zero
    n_phrases = len_batch_seqs / phrase_size # squences to phrases
    n_phrases = n_phrases.int() # ignore last phrase

    pos_start_mask = get_random_mask_phrases_idx(n_phrases, n_phrase_mask, phrase_size) # randomly mask phrases
    pos_start_mask = torch.tensor(pos_start_mask).int()

    rand_mask = torch.zeros(input.shape)
    for i in range(len(rand_mask)):
      for j in range(n_phrase_mask):       
        rand_mask[i, pos_start_mask[i,j]: pos_start_mask[i,j] + phrase_size[i]] = 1 
    return rand_mask


# def construct_att_mask(inputs, n_heads=12, vocab=None):
#     padding_mask = (inputs == vocab.index('[PAD]'))
#     mask = torch.zeros(inputs.shape)
#     mask = mask.masked_fill_(padding_mask, 1)
#     mask = mask.bool()
#     word_mask = create_attention_mask_from_input_mask(inputs, mask)#rand_mask)
#     word_mask = word_mask.bool()
#     attx_mask = torch.zeros(word_mask.shape).masked_fill_(word_mask, float('-inf'))
#     attx_mask_for_multi_head = [] 
#     for i in range(n_heads):
#         attx_mask_for_multi_head.append(attx_mask)
#     attx_mask = torch.cat(attx_mask_for_multi_head, dim=0)   # [n_head * batch_size, dim_hidden, dim_hidden]
#     return attx_mask


def mask_strategy_contiguous_split_for_HuggingBert(input, ratio=0.15, n_phrase_mask=1): 
    """
    Modified version of contiguous k-mers mask strategy;
    n_phrase_mask should be smaller than the length of sequences (in k-mers format)
    """
    def get_random_mask_phrases_idx(num_phrases, n_phrase_mask, phrase_size):
        idx_collect=[]
        for i in range(len(num_phrases)):
            # idx = random.sample([(int( j*phrase_size[i] ) + 1) for j in range(num_phrases[i])], n_phrase_mask) # add 1 for skipping [CLS]
            idx = random.sample([int( j*phrase_size[i] ) for j in range(num_phrases[i])], n_phrase_mask)             
            idx_collect.append(idx)
        return idx_collect

    # length of each sequence (in k-mer format)
    len_batch_seqs = []
    for seqKmer in input:
        len_batch_seqs.append(len(seqKmer))
    len_batch_seqs = torch.tensor(len_batch_seqs)

    phrase_size = len_batch_seqs / n_phrase_mask * ratio  # the num of words to be masked (for a phrase) 
    phrase_size = phrase_size.round().int() # rounding the numbers
    phrase_size[ phrase_size==0 ] == 1 # avoid denominator to be zero
    n_phrases = len_batch_seqs / phrase_size # squences to phrases
    n_phrases = n_phrases.int() # ignore last phrase

    pos_start_mask = get_random_mask_phrases_idx(n_phrases, n_phrase_mask, phrase_size) # randomly mask phrases
    pos_start_mask = torch.tensor(pos_start_mask).int()
   
    # add '' for transforming batch data (a list of sequences with different length) to numpy array (object type) 
    for i in range(len(input)):
        space = len_batch_seqs.max().item() - len(input[i])
        if space > 0:
           input[i] = input[i] + ['' for j in range(space)]
    added_space = np.array(input, dtype=object)     

    # replace some tokens with '[MASK]' tokens
    for i in range(len(input)):
        for j in range(n_phrase_mask):  
            added_space[i, pos_start_mask[i,j]: pos_start_mask[i,j] + phrase_size[i]] = '[MASK]'

    # for Huggindface Bert Tokenizer input format
    collect = []
    for row in added_space:
        row = remove_noneElements_in_list(row) # remove the space element ""
        collect.append(' '.join(row))

    collect_label = []
    for row in input:      
        row = remove_noneElements_in_list(row) # remove the space element ""
        collect_label.append(' '.join(row))

    return collect, collect_label


def mask_strategy_contiguous_for_HuggingBert(input, ratio=0.15): 
    """
    The strategy used in DNABert
    """
    # length of each sequence (in k-mer format)
    len_batch_seqs = []
    for seqKmer in input:
        len_batch_seqs.append(len(seqKmer))
    len_batch_seqs = torch.tensor(len_batch_seqs)

    # len_batch_seqs = torch.argwhere(input == vocab['[SEP]'])[:,1] - 1
    n_words_mask = len_batch_seqs * ratio
    n_words_mask = n_words_mask.round().int()
   
    # rand_value = torch.randint(1, int((1-ratio)*100), [len(input)]) / 100 # mask start pointer
    rand_value = torch.randint(0, int((1-ratio)*100), [len(input)]) / 100 # mask start pointer
    pos_start_mask = len_batch_seqs * rand_value
    pos_start_mask = pos_start_mask.round().int()

    # add '' for transforming batch data (a list of sequences with different length) to numpy array (object type) 
    for i in range(len(input)):
        space = len_batch_seqs.max().item() - len(input[i])
        if space > 0:
           input[i] = input[i] + ['' for j in range(len(space))]
    added_space = np.array(input, dtype=object)  
   
    # replace some tokens with '[MASK]' tokens
    for i in range(len(input)):
        added_space[i, pos_start_mask[i]: pos_start_mask[i] + n_words_mask[i]] = '[MASK]'
    
    # for Huggindface Bert Tokenizer input format
    collect = []
    for row in added_space:     
        row = remove_noneElements_in_list(row) # remove the space element ""
        collect.append(' '.join(row))

    collect_label = []
    for row in input:
        row = remove_noneElements_in_list(row) # remove the space element ""
        collect_label.append(' '.join(row))

    return collect, collect_label


def mask_for_HuggingBert(input, ratio=0.15):
    """
    Random mask words with specific ratio
    """
    # length of each sequence
    len_batch_seqs = []
    for seq in input:
        len_batch_seqs.append(len(seq))
    len_batch_seqs = torch.tensor(len_batch_seqs)

    n_words_mask = len_batch_seqs * ratio
    n_words_mask = n_words_mask.round().int()
    
    idx_collect = []
    for i in range(len(n_words_mask)):
        idx = random.sample([int(j) for j in range(len_batch_seqs[i])], n_words_mask[i])
        idx_collect.append(idx)

    # add '' for transforming batch data (a list of sequences with different length) to numpy array (object type) 
    for i in range(len(input)):
        space = len_batch_seqs.max().item() - len(input[i])
        if space > 0:
           input[i] = input[i] + ['' for j in range(space)]
    added_space = np.array(input, dtype=object)  

    # replace some tokens with '[MASK]' tokens
    for i in range(len(input)):
        for j in range(n_words_mask[i]):  
            added_space[i, idx_collect[i][j]] = '[MASK]'

    # for Huggindface Bert Tokenizer input format
    collect = []
    for row in added_space:
        row = remove_noneElements_in_list(row) # remove the space element ""
        collect.append(' '.join(row))

    collect_label = []
    for row in input:      
        row = remove_noneElements_in_list(row) # remove the space element ""
        collect_label.append(' '.join(row))

    return collect, collect_label


def get_output_from_layer(layer_num):
    def hook(model, input, output):
        print('XU: hook here~~')
    return hook


##################### HT-SELEX data processing #####################
def check_unpacked_results(files_dir, files_all_SRAid):
    collect = []
    fastq_files = list_name_inDir(files_dir)
    for file in fastq_files:
        tmp = file.split('_')
        _SRA_ID = tmp[1]
        collect.append(_SRA_ID)

    error_ID = list( set(files_all_SRAid) - set(collect) )
    if len(error_ID)>0:
        print('Error in SRA_IDs:\t',error_ID)
        return False
    else:
        print('Check done.')
        return True


def match_EnsemblID_get_infoFromUniprot(EnsemblIDs, uniprot_map):
    collect = []
    for id in EnsemblIDs:
        tmp = uniprot_map[uniprot_map['From'] == id].values.ravel()
        tmp = tmp[1:-1]
        collect.append(tmp)
    df = pd.DataFrame(collect)
    df.columns = ['Uniprot Entry ID','Uniprot Entry Name','Uniprot Protein Name']
    return df 


##################### Torch data interface for baseline models #####################
class Dataset_FOR_BASELINE_MODEL(Dataset):
    def __init__(self, sample):
        self.sample = sample
        
    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self, idx):
        return self.sample[idx]


def data_collate_fn_FOR_BASELINE_MODEL(arr):  
    batch_seqs = [] 
    batch_labels = []
    for i in arr:
        batch_seqs.append(i[0])
        batch_labels.append(i[1])
    return batch_seqs, batch_labels




##################### unclassified #####################
def clamp_value(x):
    return torch.clamp(x, .1, 1e3)


def load_data_kmer_format(data_path):    
    f = open(data_path, 'r')
    collect = []
    for line in f:
        collect.append( line.strip().split('\t') )
    f.close()
    return collect


def set_seeds(seed_val): 
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # k-mer vocabulary
    parser.add_argument('--vocab_construct', required=False, default=False, type=bool, help="Enable this term to construct k-mer vocabulary.")
    parser.add_argument('--k', required=False, default=0, type=int, help="k-mer's hyper-parameter k.")
    parser.add_argument('--base_type', required=False, default= 'DNA', type=str, help="For DNA or RNA sequences.")
    parser.add_argument('--add_wildcard', required=False, default=False, type=bool, help="Add wildcard 'N' for unkown base letter?.")
 
    # sequence -> k-mer
    parser.add_argument('--seq_2_kmer', required=False, default=False, type=bool, help="Enable this term to convert sequence to k-mer format.")
    parser.add_argument('--sequence_dir', required=False, default=' ', type=str, help="The path of sequences file(s).")
    parser.add_argument('--tokens_cls_sep', required=False, default=False, type=bool, help="Add special tokens [CLS] and [SEP] when converting sequence to k-mer?")
    # k-mer data -> filesm for easy loading in memory
    parser.add_argument('--kmer_2_file', required=False, default=False, type=bool, help="Enable this term to convert k-mer files to OneLine files. This is for easy load large-scale data in computer memory.")
    parser.add_argument('--kmer_dir', required=False, default=' ', type=str, help="The path of k-mer file(s).")
    # sequence data -> filesm for easy loading in memory
    parser.add_argument('--seq_2_file', required=False, default=False, type=bool, help="Enable this term to convert sequence files to OneLine files. This is for easy load large-scale data in computer memory.")
    parser.add_argument('--seq_dir', required=False, default=' ', type=str, help="The path of sequence file(s).")
    parser.add_argument('--chunk_size', required=False, default=None, type=int, help=".")
    parser.add_argument('--chunk_start', required=False, default=0, type=int, help=".")


    parser.add_argument('--input_label', required=False, default=False, type=bool, help="Does the input sequence file contains label?")
    parser.add_argument('--level_sub_folder', required=False, default=1, type=int, help=".")

    parser.add_argument('--count_char_types', required=False, default=False, type=bool, help="Count different char types in a file")
    
    
    parser.add_argument('--file_path', required=False, default= ' ', type=str, help="File path for specific operation")
    parser.add_argument('--save_dir', required=False, default= ' ', type=str, help="Saving path for the results.")
    args = parser.parse_args()

    ############ k-mer vocabulary ##############
    if args.vocab_construct:
        if args.k == 0:
            print('Param --k should be provided when constructing k-mer vocab')
            import sys
            sys.exit()
        if args.save_dir == ' ':
            print('Param --save_dir should be provided when constructing k-mer vocab')
            import sys
            sys.exit()

        special_tokens_for_bert = ['[MASK]', '[CLS]', '[PAD]', '[SEP]', '[UNK]']
        vocab, len_vocab = create_k_mer_vocab(args.k, base_type=args.base_type, add_unknown_base=args.add_wildcard)
        final_vocab = vocab + special_tokens_for_bert
        save_name = 'vocab_' + args.base_type + '_' + str(args.k) + '_mer.txt'
        Dir_check_and_create(args.save_dir)
        save_txt_single_column(os.path.join(args.save_dir, save_name), final_vocab)         
    # example: 
    # python utils.py --vocab_construct True --k 4 --save_dir ./vocab_k_mer --add_wildcard True
    # python utils.py --vocab_construct True --k 3 --save_dir ./vocab_k_mer 


    ########### pre-process sequences, convert them into k-mers ##############
    if args.seq_2_kmer:
        if args.k == 0:
            print('Param --k should be provided when constructing k-mer data')
            import sys
            sys.exit()
        if args.sequence_dir == ' ':
            print('Param --sequence_dir should be provided when constructing k-mer data')
            import sys
            sys.exit()
 
        pre_process_seq2kmer_files(args.sequence_dir, args.k, special_symbols=args.tokens_cls_sep, label=args.input_label, level_sub_folder=args.level_sub_folder, min_length=10)
    # example: 
    # python utils.py --seq_2_kmer True --sequence_dir /tmp/csyuxu/processed_ATAC_seq_ENCODE_p100 --k 4
    # python utils.py --seq_2_kmer True --sequence_dir /tmp/csyuxu/PRJEB3289 --k 4 --input_label True --level_sub_folder 2
    # python utils.py --seq_2_kmer True --sequence_dir /tmp/csyuxu/PRJEB14744 --k 4 --input_label True --level_sub_folder 2
    # python utils.py --seq_2_kmer True --sequence_dir /tmp/csyuxu/PRJEB9797_PRJEB20112 --k 4 --input_label True --level_sub_folder 2
    # python utils.py --seq_2_kmer True --sequence_dir /tmp/csyuxu/DREAM5_PBM_protocol --k 4 --input_label True --level_sub_folder 2
    # python utils.py --seq_2_kmer True --sequence_dir /tmp/csyuxu/packed_models_datasets/CHIP_datasets --k 4 --input_label True --level_sub_folder 2
    
    #for adapter fusion and in vivo evaluation
    # python utils.py --seq_2_kmer True --sequence_dir /tmp/csyuxu/htselex_datasets --k 4 --input_label True --level_sub_folder 2
    # python utils.py --seq_2_kmer True --sequence_dir /tmp/csyuxu/chip_seq_datasets --k 4 --input_label True --level_sub_folder 2




    ########### split sequences (k-mer format) into files (one sequence one txt file) ##############
    if args.kmer_2_file:
        if args.save_dir == ' ':
            print('Param --save_dir should be provided when constructing OneLine files')
            import sys
            sys.exit()
        if args.kmer_dir == ' ':
            print('Param --kmer_dir should be provided when constructing OneLine files')
            import sys
            sys.exit()

        split_one_seq_one_file(args.kmer_dir, args.save_dir, level_sub_folder=args.level_sub_folder, split_chunks=args.chunk_size, chunk_start=args.chunk_start)
    # example: 
    # python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/processed_ATAC_seq_ENCODE_p100_4mer --save_dir /tmp/csyuxu/processed_ATAC_seq_ENCODE_p100_4mer_pretrainDataFolder
    # python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/processed_ChIP_seq_DeepBind_4mer --save_dir /tmp/csyuxu/processed_ChIP_seq_DeepBind_4mer_finetuneDataFolder --level_sub_folder 2
    # python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/PRJEB3289_4mer --save_dir /tmp/csyuxu/PRJEB3289_4mer_finetuneDataFolder --level_sub_folder 2
    # python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/PRJEB9797_PRJEB20112_4mer --save_dir /tmp/csyuxu/PRJEB9797_PRJEB20112_4mer_finetuneDataFolder --level_sub_folder 2
    # python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/PRJEB14744_4mer --save_dir /tmp/csyuxu/PRJEB14744_4mer_finetuneDataFolder --level_sub_folder 2 --chunk_size 207 --chunk_start 621
    # python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/DREAM5_PBM_protocol_4mer --save_dir /tmp/csyuxu/DREAM5_PBM_protocol_4mer_DataFolder --level_sub_folder 2 
    # python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/packed_models_datasets/CHIP_datasets_4mer --save_dir /tmp/csyuxu/packed_models_datasets/CHIP_datasets_4mer_DataFolder --level_sub_folder 2 
    
    #for adapter fusion and in vivo evaluation
    # python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/htselex_datasets_4mer --save_dir /tmp/csyuxu/htselex_datasets_4mer_DataFolder --level_sub_folder 2 
    # python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/chip_seq_datasets_4mer --save_dir /tmp/csyuxu/chip_seq_datasets_4mer_DataFolder --level_sub_folder 2 
    
                       
    
    ########### split sequences into files (one sequence one txt file) ##############
    if args.seq_2_file:
        if args.save_dir == ' ':
            print('Param --save_dir should be provided when constructing OneLine files')
            import sys
            sys.exit()
        if args.seq_dir == ' ':
            print('Param --seq_dir should be provided when constructing OneLine files')
            import sys
            sys.exit()

        split_one_seq_one_file(args.seq_dir, args.save_dir, level_sub_folder=args.level_sub_folder)
    # example: 
    # python utils.py --seq_2_file True --seq_dir /tmp/csyuxu/processed_uniref30_sequences --save_dir /tmp/csyuxu/processed_uniref30_sequences_pretrainDataFolder
   



    # if args.count_char_types:
    #     chars, n_types = get_aminoacid_type_from_seqfile(args.file_path)
    #     special_tokens_for_bert = ['[MASK]', '[CLS]', '[PAD]', '[SEP]', '[UNK]']
    #     chars = chars + special_tokens_for_bert
    #     if args.save_dir != ' ':
    #         save_txt_single_column(args.save_dir, chars)

    # example: 
    # python utils.py --count_char_types True --file_path /tmp/csyuxu/processed_uniref30_sequences/chunk_1.seq 
    # python utils.py --count_char_types True --file_path /tmp/csyuxu/processed_uniref30_sequences/chunk_1.seq --save_dir ./vocab/vocab_amino_acid.txt


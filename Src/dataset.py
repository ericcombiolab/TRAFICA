from SeqFolder import SeqFolder
from torch.utils.data import DataLoader
import numpy as np
import sys

sys.path.append('..')
from utils import *



class TRAFICA_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def DataLoader_with_SeqFolder(data_path, batch_size, shuffle=False, max_sample=None, k=4):

    def Data_Collection_with_SeqFolder(arr):   # padding with [PAD] / tokenizing bases into numbers when loading sequence data
        """
        Data_Collection function to get sequences and labels from the k-mer matrix format 
        Args:
        arr (list): k-mer matrix. 
        k-mer matrix format (2D list):
                    col1        col2        col3
            row1    [[k-mer 1,   k-mer 2,    label],
            row2    [k-mer 1,     k-mer 2,     label]]
        Return:
        collect_x (list): 
                    col1        col2
            row1    k-mer 1     k-mer 2 (sequence1)
            row2    k-mer 1     k-mer 2 (sequence2)
        collect_y (list): corresponding labels
        """
        collect_x = []
        collect_y = []
        for k_mer in arr:
            tmp =k_mer[:-1]
            y = k_mer[-1]
            collect_x.append(' '.join(tmp))
            collect_y.append(float(y))
            
        return collect_x, collect_y
        #return Tokenizer(collect, return_tensors="pt", padding=True), collect_y

    data = SeqFolder(data_path, max_sample=max_sample) # for loading OneLine files from a folder

    if data.__len__() < batch_size: # avoid the error when the size of the dataset less than the batch size
        batch_size = data.__len__()

    dataloader = DataLoader(data, batch_size, shuffle=shuffle, collate_fn=Data_Collection_with_SeqFolder)

    return dataloader, data


def Data_Collection(arr): 
    """
    Data_Collection function to get sequences and labels from the data matrix
    Data matrix format:
                col1        col2
        row1    sequence1   label1
        row2    sequence2   label2

    Args:
    arr (numpy.array): Data matrix. 

    Return:
    collect_x (list): 
                col1        col2
        row1    k-mer 1     k-mer 2 (sequence1)
        row2    k-mer 1     k-mer 2 (sequence2)
    y (list): corresponding labels
    """
    y = None
    arr = np.array(arr)
    
    if arr.shape[1] == 2:
        X = arr[:,0]
        y = arr[:,1].astype("float")
    elif arr.shape[1] == 1:
        X = arr[:,0]

    X = seq2kmer(list(X), k=4, min_length=10) # default k=4 in the TRAFICA study
    collect_x = []
    for i in X:
        collect_x.append(' '.join(i))

    return collect_x, y


if __name__ == '__main__':
    print('Data loading functions')

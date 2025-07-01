import h5py
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class TRAFICA_Dataset_H5DF(Dataset):
    def __init__(self, hdf5_path):
        # load HDF5 file        
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.column_names = list(self.hdf5_file.keys())
        self.datasets = {col: self.hdf5_file[col] for col in self.column_names}
    
    def __len__(self):
        return self.datasets[self.column_names[0]].shape[0]
    
    def __getitem__(self, idx):
        return [self.datasets[col][idx] for col in self.column_names]

    # def __del__(self):
    #   if not self.hdf5_file.closed:
    #     self.hdf5_file.close()


def data_collection_h5df(data):
    '''
    Designed for TRAFICA ATAC-seq data format
    '''
    data = np.stack(data,axis=1) 
    df = pd.DataFrame(data).T
  

    df.columns = ['cell type', 'chrom', 'peaks_downstream', 
                  'peaks_upstream', 'sequence']
    # df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    return df


def load_dataset_from_h5(h5_path, batch_size=128, shuffle=False):
    dataset = TRAFICA_Dataset_H5DF(h5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collection_h5df)
    return dataloader



############################################# 
# TF-DNA binding profiles
# col1-sequence col2-label

class TRAFICA_Dataset_TXT(Dataset):
    def __init__(self, txt_path):
        # load txt file        
        self.seq_file = pd.read_table(txt_path, header=None)
        self.datasets = self.seq_file.values
        
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, idx):
        return self.datasets[idx] 


def data_collection_txt(arr):
    y = None
    arr = np.array(arr)
    
    if arr.shape[1] == 2:
        X = arr[:,0]
        y = arr[:,1].astype("float")
    elif arr.shape[1] == 1:
        X = arr[:,0]
    return X.tolist(), y    


def load_dataset_from_txt(txt_path, batch_size=128, shuffle=False):
    dataset = TRAFICA_Dataset_TXT(txt_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collection_txt)
    return dataloader




############################################# 
class TRAFICA_Dataset_Sequence(Dataset):
    def __init__(self, sequence_path):
        # load txt file        
        self.seq_file = pd.read_table(sequence_path, header=None)
        self.datasets = self.seq_file.values.ravel()
        
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, idx):
        return self.datasets[idx] 


def data_collection_sequence(arr):
    return arr   


def load_sequence_from_txt(sequence_path, batch_size=128, shuffle=False):
    dataset = TRAFICA_Dataset_Sequence(sequence_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collection_sequence)
    return dataloader
    
    
    
    
if __name__ == '__main__':
    ###### module 
    print('TRAFICA datasets')

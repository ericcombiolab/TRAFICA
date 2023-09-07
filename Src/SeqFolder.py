import torch.utils.data as data
import os
import sys
import numpy as np


def make_dataset(dir):
    data = []
    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)  
            data.append(path)

    #data = data[:2000]  # small subset for debug    
    #print('data size:\t', len(data))
    return data

class DatasetFolder(data.Dataset):
    def __init__(self, folder, loader, max_sample=None):
       
        samples = make_dataset(folder) # len(samples) == num of sequences

        if max_sample:
            if max_sample < len(samples):
                samples = samples[:max_sample]


        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + folder + "\n"))

        self.folder = folder
        self.loader = loader
        self.samples = samples

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        return sample

    def __len__(self):
        return len(self.samples)

    def get_sample(self): # for external usage
        return self.samples


## XU: np.loadtxt() would cause fatal error ... pointer to invalid memory location!!  Segmentation fault (core dumped). 
# def default_loader(path):
#     print(np.loadtxt(path,delimiter="\t",dtype=str).tolist()[:-1])
#     import sys
#     sys.exit()
#     return np.loadtxt(path,delimiter="\t",dtype=str).tolist()[:-1] # -1: remove space symbol
def default_loader(path):
    with open(path, 'r') as f:
        line = f.readlines()[0]
        data = line.strip().split('\t')
    return data



class SeqFolder(DatasetFolder):
    def __init__(self, folder ,loader=default_loader,  max_sample=None):
        super(SeqFolder, self).__init__(folder, loader, max_sample)
        self.seqs = self.samples



if __name__ == '__main__':
    print('For large-scale dataset')
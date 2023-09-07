import os
import sys

sys.path.append('..')
from utils import load_txt_single_column, save_txt_single_column


class Parallel_Recorder:
    def __init__(self, record_file='finished_exps.txt', task_list=None):
        self.record_file = record_file
        self.task_list = task_list
        self.unfinished_exps = None
        self.finished_exps = None 
        self._init()

    def _init(self):    
        if not os.path.exists(self.record_file):
            f = open(self.record_file,'w') # create a blank file
            f.close()
        # check unfinished experiments
        self.finished_exps = load_txt_single_column(self.record_file)
        self.unfinished_exps = list(set(self.task_list) - set(self.finished_exps))
        self.unfinished_exps.sort(key = self.task_list.index)   
    
    def update(self, current):   # before running  
        self.finished_exps.append(current)    
        save_txt_single_column(self.record_file, self.finished_exps)

    def refresh(self):    # after running, check for next run
        self.finished_exps = load_txt_single_column(self.record_file)
        self.unfinished_exps = list( set(self.task_list) - set(self.finished_exps) )
        self.unfinished_exps.sort(key=self.task_list.index)   
           

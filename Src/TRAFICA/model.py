import torch
import torch.nn as nn
from transformers import RoFormerModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

import os
import sys


sys.path.append('..')
from util import *



###### Pretrain: Mask language model
class TRAFICA_PreTrain(nn.Module):
    def __init__(self, configuration=None):
        super(TRAFICA_PreTrain, self).__init__()

        self.tranformer_model = RoFormerModel(configuration)
        self.token_predictor = BertOnlyMLMHead(configuration)
        
    def forward(self, X):
        out = self.tranformer_model(**X)
        logit = self.token_predictor(out.last_hidden_state)
        return logit, out
    

import torch
import torch.nn as nn
from transformers import BertModel, DebertaModel, AdapterConfig, AdapterFusionConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPooler
from transformers.models.deberta.modeling_deberta import DebertaOnlyMLMHead
from transformers.adapters.composition import Fuse

import os

import sys
sys.path.append('..')
from utils import *


# pretrain MLM
class Bert_MLM_model(nn.Module):
    def __init__(self, configuration=None):
        super(Bert_MLM_model, self).__init__()

        self.BertModel = BertModel(configuration, add_pooling_layer=False)
        self.BertOnlyMLMHead = BertOnlyMLMHead(configuration)
        
    def forward(self, X):
        out = self.BertModel(**X)
        logit = self.BertOnlyMLMHead(out.last_hidden_state)
        return logit, out


# modified version
class mean_BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kmers_token_tensor = hidden_states[:,1:-1,:]
        mean_token_tensor = torch.mean(kmers_token_tensor, dim=1)
        pooled_output = self.dense(mean_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

# finetune classification
class Bert_seqClassification(nn.Module):
    def __init__(self, configuration=None, model_path=None, pool_strategy='mean', fine_tuned=False):
        super(Bert_seqClassification, self).__init__()

        self.pool_strategy = pool_strategy
        self.bert_config = configuration
        self.classifier = nn.Linear(configuration.hidden_size, 1)
           
        if pool_strategy == 'mean':
            self.pooler  = mean_BertPooler(configuration) 
        elif pool_strategy == 't_cls':
            self.pooler  = BertPooler(configuration)
    
        # fine tune  or  predict 
        if fine_tuned:
            self.BertModel = BertModel(configuration, add_pooling_layer=False)              
            self.from_finetuned(model_path)
        else:
            self.BertModel = BertModel.from_pretrained(model_path, add_pooling_layer=False)       
            # self.BertModel = BertModel(configuration, add_pooling_layer=False) # evaluation without pre-train
             
    def forward(self, X):
        out = self.BertModel(**X)     
        cls_out = self.pooler(out.last_hidden_state) 
        logit = self.classifier(cls_out)
        logit = torch.sigmoid(logit)
        return logit

    def save_finetuned(self, save_dir):
        self.bert_config.to_json_file(os.path.join(save_dir, 'config.json')) # save bert configuration
        state = {'bert':self.BertModel.state_dict() , 'pooler':self.pooler.state_dict(), 'classifier':self.classifier.state_dict()}
        torch.save(obj=state, f= os.path.join(save_dir, 'finetuned_model.pth'))     
        
    def from_finetuned(self, model_path):
        state_dict = torch.load(os.path.join(model_path, 'finetuned_model.pth'))
        self.BertModel.load_state_dict(state_dict['bert'])
        self.classifier.load_state_dict(state_dict['classifier'])
        self.pooler.load_state_dict(state_dict['pooler'])


# finetune regressor
class Bert_seqRegression(nn.Module):
    def __init__(self, configuration=None, model_path=None, pool_strategy='mean', fine_tuned=False, out_attention=False):
        super(Bert_seqRegression, self).__init__()


        self.pool_strategy = pool_strategy
        self.out_attention = out_attention
        self.regresstor = nn.Sequential(nn.Linear(configuration.hidden_size, int(configuration.hidden_size/2)), nn.Linear(int(configuration.hidden_size/2), 1))
 
        if pool_strategy == 'mean':
            self.pooler  = mean_BertPooler(configuration) 
        elif pool_strategy == 't_cls':
            self.pooler  = BertPooler(configuration)
    
        # fine tune  or  predict 
        if fine_tuned:
            self.BertModel = BertModel(configuration, add_pooling_layer=False)
            self.from_finetuned(model_path)
        else:
            self.BertModel = BertModel.from_pretrained(model_path, add_pooling_layer=False)
           # self.BertModel = BertModel(configuration, add_pooling_layer=False) # evaluation without pre-train
       

    def forward(self, X):
        out = self.BertModel(**X)
        cls_out = self.pooler(out.last_hidden_state) 
        logit = self.regresstor(cls_out) 
        if self.out_attention:
            return logit, out.attentions
        else:
            return logit 

    def save_finetuned(self, save_dir):
        self.BertModel.config.to_json_file(os.path.join(save_dir, 'config.json')) # save bert configuration
        state = {'bert':self.BertModel.state_dict() , 'pooler':self.pooler.state_dict(),'regresstor':self.regresstor.state_dict()}  
        torch.save(obj=state, f= os.path.join(save_dir, 'finetuned_model.pth'))     
        
    def from_finetuned(self, model_path):
        state_dict = torch.load(os.path.join(model_path, 'finetuned_model.pth'))

        self.BertModel.load_state_dict(state_dict['bert'])
        self.regresstor.load_state_dict(state_dict['regresstor'])
        self.pooler.load_state_dict(state_dict['pooler'])


# class Bert_seqRegression(nn.Module):
#     def __init__(self, configuration=None, model_path=None, pool_strategy='mean', fine_tuned=False, out_attention=False):
#         super(Bert_seqRegression, self).__init__()

#         self.pool_strategy = pool_strategy
#         self.out_attention = out_attention
#         # fine tune  or  predict 
#         if fine_tuned:
#             self.BertModel = BertModel(configuration, add_pooling_layer=False)
#             h_size = self.BertModel.config.hidden_size
#             self.regresstor = nn.Sequential(nn.Linear(h_size, int(h_size/2)), nn.Linear(int(h_size/2), 1))
#             self.from_finetuned(model_path)
#         else:
#             self.bert_config = configuration
#             self.BertModel = BertModel.from_pretrained(model_path, add_pooling_layer=False)
#             # self.BertModel = BertModel(configuration, add_pooling_layer=False) # evaluation without pre-train
#             h_size = self.BertModel.config.hidden_size
#             self.regresstor = nn.Sequential(nn.Linear(h_size, int(h_size/2)), nn.Linear(int(h_size/2), 1))


#     def forward(self, X):
#         out = self.BertModel(**X)
#         if self.pool_strategy == 't_cls': ## token [CLS]
#             cls_out = out.last_hidden_state[:,0]
#         elif self.pool_strategy == 'mean': ## pooling strategy: mean all k-mer embedding     
#             cls_out = out.last_hidden_state[:,1:-1,:]
#             cls_out = torch.mean(cls_out, dim=1)
#         logit = self.regresstor(cls_out) 

#         if self.out_attention:
#             return logit, out.attentions
#         else:
#             return logit 

#     def save_finetuned(self, save_dir):
#         self.bert_config.to_json_file(os.path.join(save_dir, 'config.json')) # save bert configuration
#         state = {'bert':self.BertModel.state_dict() , 'regresstor':self.regresstor.state_dict()}  
#         torch.save(obj=state, f= os.path.join(save_dir, 'finetuned_model.pth'))     
        
#     def from_finetuned(self, model_path):
#         state_dict = torch.load(os.path.join(model_path, 'finetuned_model.pth'))
#         self.BertModel.load_state_dict(state_dict['bert'])
#         self.regresstor.load_state_dict(state_dict['regresstor'])



# finetune regresstor with adapters
class Bert_seqRegression_adapter(nn.Module):
    def __init__(self, pretrain_modelpath=None, model_path=None, pool_strategy='mean', fine_tuned=False, out_attention=False, reduction_factor=16):
        super(Bert_seqRegression_adapter, self).__init__()

        self.pool_strategy = pool_strategy
        self.out_attention = out_attention

        # load pretrained model
        self.BertModel = BertModel.from_pretrained(pretrain_modelpath, add_pooling_layer=False)

        # add pool layer
        if pool_strategy == 'mean':
            self.pooler  = mean_BertPooler(self.BertModel.config) 
        elif pool_strategy == 't_cls':
            self.pooler  = BertPooler(self.BertModel.config)

        # add regression layer
        h_size = self.BertModel.config.hidden_size
        self.regresstor = nn.Sequential(nn.Linear(h_size, int(h_size/2)), nn.Linear(int(h_size/2), 1))
   
        # fine tune  or  predict 
        if fine_tuned:                  
            self.from_finetuned(model_path)
        else:
            adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)
            self.BertModel.add_adapter("SingleAdapter", config=adapter_config)
            self.BertModel.train_adapter("SingleAdapter")
        
    def forward(self, X):
        out = self.BertModel(**X)     
        cls_out = self.pooler(out.last_hidden_state) 
        logit = self.regresstor(cls_out) 

        if self.out_attention:
            return logit, out.attentions
        else:
            return logit 
        


    def save_finetuned(self, save_dir):
        # only save the regresstor and the adapter parameters
        regresstor_path = os.path.join(save_dir, 'regresstor')
        create_directory(regresstor_path)
        state = {'regresstor':self.regresstor.state_dict(), 'pooler':self.pooler.state_dict()}  
        torch.save(obj=state, f= os.path.join(regresstor_path, 'finetuned_model.pth'))  

        adapter_path = os.path.join(save_dir, 'adapter')
        create_directory(adapter_path)
        self.BertModel.save_adapter(save_directory=adapter_path, adapter_name='SingleAdapter')


    def from_finetuned(self, model_path):
        regresstor_path = os.path.join(model_path, 'regresstor')
        adapter_path = os.path.join(model_path, 'adapter')
        state_dict = torch.load(os.path.join(regresstor_path, 'finetuned_model.pth'))

        self.BertModel.load_adapter(adapter_name_or_path=adapter_path, load_as='SingleAdapter', set_active=True)  ## use "load_as" to rename the adapter
        self.regresstor.load_state_dict(state_dict['regresstor'])
        self.pooler.load_state_dict(state_dict['pooler'])



# finetune regresstor with adapters
class Bert_seqClassification_adapter(nn.Module):
    def __init__(self, pretrain_modelpath=None, model_path=None, pool_strategy='mean', fine_tuned=False, out_attention=False, reduction_factor=16):
        super(Bert_seqClassification_adapter, self).__init__()

        self.pool_strategy = pool_strategy
        self.out_attention = out_attention
        self.BertModel = BertModel.from_pretrained(pretrain_modelpath, add_pooling_layer=False)

        h_size = self.BertModel.config.hidden_size
        # self.bert_config = configuration
        self.classifier = nn.Linear(h_size, 1)
           
        if pool_strategy == 'mean':
            self.pooler  = mean_BertPooler(self.BertModel.config) 
        elif pool_strategy == 't_cls':
            self.pooler  = BertPooler(self.BertModel.config)

        # fine tune  or  predict 
        if fine_tuned:           
            self.from_finetuned(model_path)
        else:
            adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)
            self.BertModel.add_adapter("SingleAdapter", config=adapter_config)
            self.BertModel.train_adapter("SingleAdapter")


    def forward(self, X):
        out = self.BertModel(**X)     
        cls_out = self.pooler(out.last_hidden_state) 
        logit = self.classifier(cls_out)
        logit = torch.sigmoid(logit)

        if self.out_attention:
            return logit, out.attentions
        else:
            return logit 

     
    def save_finetuned(self, save_dir):
        # only save the regresstor and the adapter parameters
        classifier_path = os.path.join(save_dir, 'classifier')
        create_directory(classifier_path)
        state = {'classifier':self.classifier.state_dict(), 'pooler':self.pooler.state_dict()}  
        torch.save(obj=state, f= os.path.join(classifier_path, 'finetuned_model.pth'))  

        adapter_path = os.path.join(save_dir, 'adapter')
        create_directory(adapter_path)
        self.BertModel.save_adapter(save_directory=adapter_path, adapter_name='SingleAdapter')


    def from_finetuned(self, model_path):
        classifier_path = os.path.join(model_path, 'classifier')
        adapter_path = os.path.join(model_path, 'adapter')
        state_dict = torch.load(os.path.join(classifier_path, 'finetuned_model.pth'))

        self.BertModel.load_adapter(adapter_name_or_path=adapter_path)
        self.BertModel.set_active_adapters('SingleAdapter')

        self.classifier.load_state_dict(state_dict['classifier'])
        self.pooler.load_state_dict(state_dict['pooler'])

     



class Bert_seqRegression_adapterfusion(nn.Module):
    def __init__(self, pretrain_modelpath=None, pretrain_adapterpath=None, model_path=None, pool_strategy='mean', fine_tuned=False, out_attention=False):
        super(Bert_seqRegression_adapterfusion, self).__init__()


        self.pool_strategy = pool_strategy
        self.out_attention = out_attention

        # load pretrained model 
        self.BertModel = BertModel.from_pretrained(pretrain_modelpath, add_pooling_layer=False)
 
        # load pretrained adapters 
        trained_adapters = list(pretrain_adapterpath.keys())

        for i in trained_adapters:
            self.BertModel.load_adapter(adapter_name_or_path=os.path.join(pretrain_adapterpath[i], 'adapter'), load_as=i, set_active=True)

        # self.adapters_list = trained_adapters

        # add pool layer
        if pool_strategy == 'mean':
            self.pooler  = mean_BertPooler(self.BertModel.config) 
        elif pool_strategy == 't_cls':
            self.pooler  = BertPooler(self.BertModel.config)

        # add regression layer
        h_size = self.BertModel.config.hidden_size
        self.regresstor = nn.Sequential(nn.Linear(h_size, int(h_size/2)), nn.Linear(int(h_size/2), 1))
   

        # fine tune  or  predict 
        if fine_tuned:                  
            self.from_finetuned(model_path)
        else:           
            # # load pooler and predictor's parameters
            # state_dict = torch.load(os.path.join(pretrain_adapterpath[trained_adapters[0]], 'regresstor', 'finetuned_model.pth'))
            # self.regresstor.load_state_dict(state_dict['regresstor'])
            # self.pooler.load_state_dict(state_dict['pooler'])
            # # froze params
            # for param in self.regresstor.parameters():
            #     param.requires_grad = False
            # for param in self.pooler.dense.parameters():
            #     param.requires_grad = False


            # add fusion module and enable it for training
            self.fusion_setup = Fuse(*trained_adapters)    
            self.BertModel.add_adapter_fusion(self.fusion_setup, set_active=True)
            self.BertModel.train_adapter_fusion(self.fusion_setup)


    def forward(self, X):  

        out = self.BertModel(**X)     
        cls_out = self.pooler(out.last_hidden_state) 
        logit = self.regresstor(cls_out) 

        if self.out_attention:
            return logit, out.attentions
        else:
            return logit 
        

    def from_finetuned(self, model_path):
        regresstor_path = os.path.join(model_path, 'regresstor')
        adapter_path = os.path.join(model_path, 'adapter')
        state_dict = torch.load(os.path.join(regresstor_path, 'finetuned_model.pth'))
 
        self.BertModel.load_adapter_fusion(adapter_path, set_active=True)
        self.regresstor.load_state_dict(state_dict['regresstor'])
        self.pooler.load_state_dict(state_dict['pooler'])

    
    def save_finetuned(self, save_dir):
        # only save the regresstor and the adapter parameters
        regresstor_path = os.path.join(save_dir, 'regresstor')
        create_directory(regresstor_path)
        state = {'regresstor':self.regresstor.state_dict(), 'pooler':self.pooler.state_dict()}  
        torch.save(obj=state, f= os.path.join(regresstor_path, 'finetuned_model.pth'))  

        adapter_path = os.path.join(save_dir, 'adapter')
        create_directory(adapter_path)
        self.BertModel.save_adapter_fusion(save_directory=adapter_path, adapter_names=self.fusion_setup)
        # self.BertModel.save_all_adapters(save_directory=adapter_path) # reduce the memory consumption

    
     

if __name__ == '__main__':
    print('model structures')

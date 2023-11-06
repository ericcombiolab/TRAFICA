import torch
import torch.nn as nn
from transformers import BertModel, PreTrainedModel, AdapterConfig, PretrainedConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.adapters.composition import Fuse

import os
import sys
sys.path.append('..')
from utils import *



############################################################   Refinement:XU TFAFICA code v2  ######################################################################################
###### Pretrain: Mask language model
class TRAFICA_PreTrain(nn.Module):
    def __init__(self, configuration=None):
        super(TRAFICA_PreTrain, self).__init__()

        self.prediction_model = BertModel(configuration, add_pooling_layer=False)
        self.token_predictor = BertOnlyMLMHead(configuration)
        
    def forward(self, X):
        out = self.prediction_model(**X)
        logit = self.token_predictor(out.last_hidden_state)
        return logit, out


###### Basic Modules
class TRAFICA_Affinity_predictor(nn.Module):
    def __init__(self, h_size, predict_type='regression'):
        super(TRAFICA_Affinity_predictor, self).__init__()

        self.predict_type = predict_type

        if not isinstance(h_size, int):
            raise ValueError("The hidden size of the affinity predictor is not int type")

        if predict_type == 'regression': # regressor 
            self.predictor = nn.Sequential(nn.Linear(h_size, int(h_size/2)), nn.Linear(int(h_size/2), 1))
        elif predict_type == 'classification': # classifier
            self.predictor = nn.Linear(h_size,1)
        else:
            raise TypeError("The hidden size of the affinity predictor should be regression or classification")

    def forward(self, X):
        if self.predict_type == 'classification':
            return torch.sigmoid(self.predictor(X))
        else:
            return self.predictor(X)
    


class TRAFICA_Pooler(nn.Module):
    def __init__(self, h_size):
        super(TRAFICA_Pooler, self).__init__()

        if not isinstance(h_size, int):
            raise ValueError("The hidden size of the affinity predictor is not int type")

        self.dense = nn.Linear(h_size, h_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kmers_token_tensors = hidden_states[:,1:-1,:] # discard [CLS] and [SEP]
        mean_token_tensor = torch.mean(kmers_token_tensors, dim=1)
        pooled_output = self.activation( self.dense(mean_token_tensor) )
        return pooled_output
    
    
     

class TRAFICA_peripherals_config(PretrainedConfig):
    def __init__(self,model_type:str='TRAFICA_peripherals',hidden_size:int=1024, predictor_type='regression', pool_strategy='mean', add_pooler=True, **kwargs):
        """
        TRAFICA_peripherals configuration object

        Args:
        h_size (int): The size of the hidden layer in the transformer-encoder block.  
        predictor_type (str, optional): Classification or regression.
        pool_strategy (str, optional): Not provided to users, TRAFICA currently only apply the mean tokens method.
        """
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.predictor_type = predictor_type 
        self.pool_strategy = pool_strategy
        self.add_pooler= add_pooler
        super().__init__(**kwargs)



class TRAFICA_peripherals(PreTrainedModel):

    config_class = TRAFICA_peripherals_config # fixed format to custom a HuggingFace object

    def __init__(self, config):
        super(TRAFICA_peripherals, self).__init__(config)
        """
        TRAFICA_peripherals object: consisting of a Pooler and an affinity predictor

        Args:
        config (int): TRAFICA_peripherals_config object 
        """
        # pooler 
        self.add_pooler = config.add_pooler
        if self.add_pooler != True:
            self.pooler = None
        else:
            if config.pool_strategy != 'mean':
                raise TypeError('TRAFICA current only apply the mean tokens method')
            self.pooler  = TRAFICA_Pooler(config.hidden_size)

        # affinity predictor
        self.affinity_predictor = TRAFICA_Affinity_predictor(config.hidden_size, predict_type=config.predictor_type)
 
        self.init_weights() # for the utilization of .from_pretrained() 

    def _init_weights(self, module):
        pass

    def init_weights(self):
        self.apply(self._init_weights)
     
    def forward(self, X): # takes as inputs the outputs of last transformer-encoder block
        if self.add_pooler == True:
            X = self.pooler(X)
        return self.affinity_predictor(X)
    


## Fully fine-tuning approach
class TRAFICA_FullyFineTuning(nn.Module):
    def __init__(self, out_attention=False, pool_strategy='mean'):
        super(TRAFICA_FullyFineTuning, self).__init__()
        """
        TRAFICA_FullyFineTuning object: Fully mode

        Args:
        out_attention (bool, optional): The option to obtain attention matrices from all transformer-encoder blocks. 
        """
        # options to open/close functions
        self.out_attention = out_attention

        # empty when initializing a TRAFICA_Adapter object
        self.ATAC_Model = None
        self.hidden_size = None
        self.pooler_predictor = None
        self.add_pooler = None

    def forward(self, X):  
        # all components equiped ready 
        if self.pooler_predictor == None:
            raise NotImplementedError('FullyFineTuning: Did not add the pooler and the affinity predictor -> cannot forward propagation to predict affinities.')    
 
        # forward propagation
        out = self.ATAC_Model(**X)   

        if self.add_pooler:
            logit = self.pooler_predictor(out.last_hidden_state) 
        else:
            h_tokens = out.last_hidden_state[:,1:-1,:]
            h_sequence = torch.mean(h_tokens, dim=1)  
            logit = self.pooler_predictor(h_sequence) 

        # output attention
        if self.out_attention:
            return logit, out.attentions
        else:
            return logit 
        

    def init_component(self, PretrainedModelPath, add_pooler=True):
        # initialize transformer-encoder blocks -> load ATAC-seq pre-trained model 
        self.ATAC_Model = BertModel.from_pretrained(PretrainedModelPath, add_pooling_layer=False)
        self.hidden_size = self.ATAC_Model.config.hidden_size

        # initialize periphrals
        self.add_pooler = add_pooler
        peripherals_config = TRAFICA_peripherals_config(hidden_size=self.hidden_size, add_pooler=add_pooler)
        self.pooler_predictor = TRAFICA_peripherals(peripherals_config)


    def from_finetuned(self, FineTunedPath):
        finetuned_encoderblocks =  os.path.join(FineTunedPath,'fullytuned')
        finetuned_peripherals = os.path.join(FineTunedPath,'peripherals')
        self.ATAC_Model = BertModel.from_pretrained(finetuned_encoderblocks, add_pooling_layer=False)  
        self.pooler_predictor = TRAFICA_peripherals.from_pretrained(finetuned_peripherals)
        # check whether the fine-tuned model including a pooler module or not
        for name, _ in self.pooler_predictor.named_parameters():
            if name == 'pooler.dense.weight': 
                self.add_pooler = True


    def save_finetuned(self, SaveDir):
        if self.pooler_predictor == None:
            raise NotImplementedError('FullyFineTuning: Did not add the pooler and the affinity predictor -> Nothing to save.')    
        
        # create save dir
        finetuned_encoderblocks =  os.path.join(SaveDir,'fullytuned')
        finetuned_peripherals = os.path.join(SaveDir,'peripherals')
        create_directory(finetuned_encoderblocks)
        create_directory(finetuned_peripherals)

        # save the parameters of encoder blocks and peripherals
        self.ATAC_Model.save_pretrained(finetuned_encoderblocks)
        self.pooler_predictor.save_pretrained(finetuned_peripherals)


## Adapter-tuning approach
class TRAFICA_AdapterTuning(nn.Module):
    def __init__(self, PretrainedModelPath, out_attention=False, pool_strategy='mean'):
        super(TRAFICA_AdapterTuning, self).__init__()
        """
        TRAFICA_AdapterTuning object: Adapter mode

        Args:
        PretrainedModelPath (str): The path of the pre-trained model (Chromatin Accessibility from ATAC-seq) 
        out_attention (bool, optional): The option to obtain attention matrices from all transformer-encoder blocks 
        """
        # options to open/close functions
        self.out_attention = out_attention

        # load ATAC-seq pre-trained model (transformer-encoder blocks)
        self.ATAC_Model = BertModel.from_pretrained(PretrainedModelPath, add_pooling_layer=False)
        self.hidden_size = self.ATAC_Model.config.hidden_size

        # empty when initializing a TRAFICA_Adapter object
        self.pooler_predictor = None


    def forward(self, X):  
        # all components equiped ready 
        if self.pooler_predictor == None: 
            raise NotImplementedError('AdapterTuning: Did not add the Adapter -> cannot forward propagation to predict affinities.')    
 
        # forward propagation
        out = self.ATAC_Model(**X)     
        logit = self.pooler_predictor(out.last_hidden_state) 

        # output attention
        if self.out_attention:
            return logit, out.attentions
        else:
            return logit 
        

    def init_component(self, reduction_factor, predictor_type='regression'):
        # initialize an Adapter
        adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)
        self.ATAC_Model.add_adapter("SingleAdapter", config=adapter_config)
        self.ATAC_Model.train_adapter("SingleAdapter")

        # initialize periphrals
        peripherals_config = TRAFICA_peripherals_config(hidden_size=self.hidden_size, predictor_type=predictor_type)
        self.pooler_predictor = TRAFICA_peripherals(peripherals_config)


    def from_finetuned(self, AdapterPath):
        finetuned_adapter =  os.path.join(AdapterPath,'adapter')
        finetuned_peripherals = os.path.join(AdapterPath,'peripherals')
        self.ATAC_Model.load_adapter(adapter_name_or_path=finetuned_adapter, load_as='SingleAdapter', set_active=True)  ## use "load_as" to rename the adapter
        self.pooler_predictor = TRAFICA_peripherals.from_pretrained(finetuned_peripherals)
 

    def save_finetuned(self, SaveDir):
        if self.pooler_predictor == None:
            raise NotImplementedError('Adapter: Did not add the Adapter -> Nothing to save.')    
        
        # create save dir
        finetuned_adapter =  os.path.join(SaveDir,'adapter')
        finetuned_peripherals = os.path.join(SaveDir,'peripherals')
        create_directory(finetuned_adapter)
        create_directory(finetuned_peripherals)

        # save the parameters of adapter and peripherals
        self.ATAC_Model.save_adapter(save_directory=finetuned_adapter, adapter_name='SingleAdapter')
        self.pooler_predictor.save_pretrained(finetuned_peripherals)


## AdapterFusion approach
class TRAFICA_AdapterFusion(nn.Module):
    def __init__(self, PretrainedModelPath, AdapterPath, out_attention=False, pool_strategy='mean'):
        super(TRAFICA_AdapterFusion, self).__init__()
        """
        TRAFICA_AdapterFusion object: AdapterFusion mode

        Args:
        PretrainedModelPath (str): The path of the pre-trained model (Chromatin Accessibility from ATAC-seq) 
        AdapterPath (str): The path of the fine-tuned adapters 
        out_attention (bool, optional): The option to obtain attention matrices from all transformer-encoder blocks 
        """
        # options to open/close functions
        self.out_attention = out_attention

        # load ATAC-seq pre-trained model (transformer-encoder blocks)
        self.ATAC_Model = BertModel.from_pretrained(PretrainedModelPath, add_pooling_layer=False)
        self.hidden_size = self.ATAC_Model.config.hidden_size

        # load fine-tuned Adapters
        if get_file_extension(AdapterPath) == 'json':
            with open(AdapterPath, "r") as f:
                Adapters_Dict = json.load(f)
            self.Adapters_names = list(Adapters_Dict.keys())
            for name in self.Adapters_names:
               
                self.ATAC_Model.load_adapter(adapter_name_or_path=os.path.join(Adapters_Dict[name], 'adapter'), load_as=name, set_active=True)
        
        ## TODO: to support other format. A list of adapters; Manually specifiy the name of adapters...     
        else:
            raise NotImplementedError('AdapterFusion: error in loading adapters, check the input format.')    

        # empty when initializing a TRAFICA_AdapterFusion object
        self.pooler_predictor = None


    def forward(self, X):  
        # all components equiped ready 
        if self.pooler_predictor == None:
            raise TypeError('AdapterFusion: Did not add the AdapterFusion component -> cannot forward propagation to predict affinities.')    
 
        # forward propagation
        out = self.ATAC_Model(**X)     
        logit = self.pooler_predictor(out.last_hidden_state) 

        # output attention
        if self.out_attention:
            return logit, out.attentions
        else:
            return logit 
        

    def init_component(self, predictor_type='regression'):
        # initialize an AdapterFusion component 
        self.fusion_setup = Fuse(*self.Adapters_names)   
        self.ATAC_Model.add_adapter_fusion(self.fusion_setup, set_active=True)
        self.ATAC_Model.train_adapter_fusion(self.fusion_setup)

        # initialize periphrals
        peripherals_config = TRAFICA_peripherals_config(hidden_size=self.hidden_size, predictor_type=predictor_type)
        self.pooler_predictor = TRAFICA_peripherals(peripherals_config)


    def from_finetuned(self, AdapterFusionPath):
        finetuned_adapterfusion =  os.path.join(AdapterFusionPath,'adapterfusion')
        finetuned_peripherals = os.path.join(AdapterFusionPath,'peripherals')
        self.ATAC_Model.load_adapter_fusion(finetuned_adapterfusion, set_active=True)
        self.pooler_predictor = TRAFICA_peripherals.from_pretrained(finetuned_peripherals)


    def save_finetuned(self, SaveDir):
        if self.pooler_predictor == None:
            raise NotImplementedError('AdapterFusion: Did not add the AdapterFusion component -> Nothing to save.')    
        
        # create save dir
        finetuned_adapterfusion =  os.path.join(SaveDir,'adapterfusion')
        finetuned_peripherals = os.path.join(SaveDir,'peripherals')
        create_directory(finetuned_adapterfusion)
        create_directory(finetuned_peripherals)

        # save the parameters of adapterfusion and peripherals
        self.fusion_setup = Fuse(*self.Adapters_names)   
        self.ATAC_Model.save_adapter_fusion(save_directory=finetuned_adapterfusion, adapter_names=self.fusion_setup)
        self.pooler_predictor.save_pretrained(finetuned_peripherals)
        # # self.ATAC_Model.save_all_adapters(save_directory=adapter_path) # reduce the memory consumption



class TRAFICA(nn.Module):
    def __init__(self, PreTrained_ModelPath=None, FineTuned_FullModelPath=None, FineTuned_AdapeterPath=None, FineTuned_AdapetersListPath=None, 
                 FineTuned_AdapeterFusionPath=None ,FineTuningType='AdapterTuning', PredictorType='regression', out_attention=False):
        super(TRAFICA, self).__init__()
        """
        TRAFICA object: Main body

        Args:
        PretrainedModelPath (str, optional): The path of the pre-trained model (Chromatin Accessibility from ATAC-seq); It is optional in evaluating the model in fully fine-tuning mode
        FineTuned_FullModelPath (str,optional): The path of the fully fine-tuned model
        FineTuned_AdapeterPath (str, optional): The path of the fine-tuned adapter for evaluating the model in Adapter-tuning mode
        FineTuned_AdapetersListPath (str, optional): The path of the fine-tuned adapters list for training and evaluating the model in AdapterFusion mode
        FineTuned_AdapeterFusionPath (str, optional): The path of the fine-tuned adapterfusion component for evaluating the model in AdapterFusion mode
        FineTuningType (str): TRAFICA support three types of fine-tuning: FullyFineTuning, AdapterTuning, and AdapterFusion
        PredictorType (str):TRAFICA support two types of prediction: regression and classification
        out_attention (bool, optional): The option to obtain attention matrices from all transformer-encoder blocks 
        """
        ## 
        self.out_attention = out_attention
        self.FineTuningType = FineTuningType
        self.PredictorType = PredictorType
        self.PreTrained_ModelPath = PreTrained_ModelPath
        self.FineTuned_FullModelPath = FineTuned_FullModelPath
        self.FineTuned_AdapeterPath = FineTuned_AdapeterPath 
        self.FineTuned_AdapetersListPath = FineTuned_AdapetersListPath
        self.FineTuned_AdapeterFusionPath = FineTuned_AdapeterFusionPath 

        ## initialize a model instance
        if FineTuningType == 'FullyFineTuning':
            self.prediction_model = TRAFICA_FullyFineTuning(out_attention=out_attention)
        elif FineTuningType == 'AdapterTuning' or FineTuningType == 'AdapterFusion':
            # check the input path of the pre-trained model; AdapterTuning and AdapterFusion rely on the pre-trained model in both training and prediction
            if PreTrained_ModelPath == None:
                raise ValueError('Selecting AapterTuning type: requiring the pre-trained model; specifiy the path of the model PreTrained_ModelPath=XX')
   
            if FineTuningType == 'AdapterTuning':
                self.prediction_model = TRAFICA_AdapterTuning(PretrainedModelPath=PreTrained_ModelPath, out_attention=out_attention)
            elif FineTuningType == 'AdapterFusion':
                self.prediction_model = TRAFICA_AdapterFusion(PretrainedModelPath=PreTrained_ModelPath, AdapterPath=FineTuned_AdapetersListPath, out_attention=out_attention)
        else:
            raise TypeError("TRAFICA initialization: Check FineTuningType. TRAFICA support three types of fine-tuning: FullyFineTuning, AdapterTuning, and AdapterFusion")


    def _train(self):
        if self.FineTuningType == 'FullyFineTuning':
            self.prediction_model.init_component(PretrainedModelPath=self.PreTrained_ModelPath)
        elif self.FineTuningType == 'AdapterTuning':
            self.prediction_model.init_component(reduction_factor=16, predictor_type=self.PredictorType) # 16 by default in our study
        elif self.FineTuningType == 'AdapterFusion':
            self.prediction_model.init_component(predictor_type=self.PredictorType)

        self.prediction_model.train()


    def _eval(self):
        if self.FineTuningType == 'FullyFineTuning':
            self.prediction_model.from_finetuned(self.FineTuned_FullModelPath)
        elif self.FineTuningType == 'AdapterTuning':
            self.prediction_model.from_finetuned(self.FineTuned_AdapeterPath)
        elif self.FineTuningType == 'AdapterFusion':
            self.prediction_model.from_finetuned(self.FineTuned_AdapeterFusionPath)

        self.prediction_model.eval()


    def forward(self, X):
        out = self.prediction_model(X)

        return out 





if __name__ == '__main__':
    print('model structures')

# transformers and PyTorch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch

# others
# import numpy as np
import sys
import sklearn.metrics as metrics
from scipy.stats import pearsonr
import argparse
import h5py

# TRAFICA
from models import TRAFICA
from dataset import DataLoader_with_SeqFolder, Data_Collection, TRAFICA_Dataset


sys.path.append('..')
from utils import *


def test_model(model, test_data, Tokenizer, device):
    collect_y = []
    collect_out = []
    for data in test_data:

        X, y = data
        inputs = Tokenizer(X, return_tensors="pt", padding=True)
        y = torch.tensor(data[1]).float()

        out = model(inputs.to(device))

        collect_y += y.tolist()
        collect_out += out.flatten().cpu().detach().numpy().tolist()

    return  collect_y, collect_out


def TRAFICA_HOOK_GET_Q_K_AdapterFusion(model, q_out, k_out, encoder_layer=0):
    ## hook to obtain the output of specific layers 
    def hook_forward_function_query(module, module_in, module_out):
        q_out.append(module_out.clone().detach())

    def hook_forward_function_key(module, module_in, module_out):
        k_out.append(module_out.clone().detach())

    # get the name of the adapterfusio component
    tmp_adapterfusion_dict_key = model.BertModel.encoder.layer[1].output.adapter_fusion_layer.keys()
    tmp_adapterfusion_dict_key = list(tmp_adapterfusion_dict_key)[0]

    # register hook function for callback
    hook_query = model.BertModel.encoder.layer[encoder_layer].output.adapter_fusion_layer[tmp_adapterfusion_dict_key].query.register_forward_hook(hook_forward_function_query)
    hook_key = model.BertModel.encoder.layer[encoder_layer].output.adapter_fusion_layer[tmp_adapterfusion_dict_key].key.register_forward_hook(hook_forward_function_key)
    return hook_query, hook_key


def test_model_outFusionAtt(model, test_data, Tokenizer, device):
    collect_y = []
    collect_out = []

    n_layers = len(model.BertModel.encoder.layer)
    collect_attAdapterFusion = [[] for i in range(n_layers)]

    for data in test_data:
        X, y = data
        inputs = Tokenizer(X, return_tensors="pt", padding=True)
        y = torch.tensor(data[1]).float()

        q_out_collect = [] # hiddden saving
        k_out_collect = [] # hiddden saving
        q_hook_collect = [] 
        k_hook_collect = []
        for i in range(n_layers):
            q_hook, k_hook = TRAFICA_HOOK_GET_Q_K_AdapterFusion(model, q_out_collect, k_out_collect, encoder_layer=i)
            q_hook_collect.append(q_hook)
            k_hook_collect.append(k_hook)


        out = model(inputs.to(device)) # model forward

        for i in range(n_layers):
            # attention calculation
            attMatrix_adapterfusion = torch.einsum('bij,bijk->bik', q_out_collect[i] , k_out_collect[i].transpose(-1, -2)) # (batch, seq_length, n_adapeters) = # (batch, seq_length, h_dim)  dot-product (batch, seq_length, h_dim, n_adapeters)     
            attMatrix_adapterfusion = torch.softmax(attMatrix_adapterfusion,dim=-1) # (batch, seq_length, n_adapeters); for each adapter contribute to each token
            attScore_adapterfusion = torch.sum(attMatrix_adapterfusion,dim=1) # (batch, n_adapeters) for each adapter to all tokens
            # collect the info of each layer
            collect_attAdapterFusion[i].append(attScore_adapterfusion.cpu()) 
            # release hook handle
            q_hook_collect[i].remove()
            k_hook_collect[i].remove()

        collect_y += y.tolist()
        collect_out += out.flatten().cpu().detach().numpy().tolist()

    ## collect batch hidden features for all transformer-encode layers 
    collect_attAdapterFusion_all = []
    for i in range(n_layers):
        collect_attAdapterFusion_all.append(torch.cat(collect_attAdapterFusion[i],dim=0))
  
    return  collect_y, collect_out, collect_attAdapterFusion_all




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic setting
    parser.add_argument('--batch_size', required=False, type=int, default=1, help="Batch size for evaluation. Default:1")
    parser.add_argument('--use_gpu', default=False, type=bool, help="Utilize gpu to accelerate inference procedure; False: using cpu, True: using gpu if available.")
    parser.add_argument('--task_type', default='AdapterTuning', type=str, help="TRAFICA support three types of fine-tuning: FullyFineTuning, AdapterTuning, and AdapterFusion")
    parser.add_argument('--evaluation_metric', required=True, default='set1', type=str, help="set1: auroc; set2: pcc & r2")
    parser.add_argument('--out_att_adapters', default=False, type=bool, help="output attention score of adapters in the AdapterFusion phrase")
    parser.add_argument('--inputfile_types', default='DataMatrix', type=str, help="OneLine file or DataMatrix")
    # path
    parser.add_argument('--save_dir', required=True,type=str, help="The saving path of the prediction result")
    parser.add_argument('--data_path', required=True, type=str, help="The path of the data folder")
    parser.add_argument('--pretrain_tokenizer_path', required=True, type=str, help="The path of the tokenizer (4-mer)")
    parser.add_argument('--pretrained_model_path', required=False, default=None, type=str, help="The path of the pretrained model")
    parser.add_argument('--finetuned_fullmodel_path', required=False, default=None, type=str, help="The path of the fully finetuned model (Fully fine-tuning)")
    parser.add_argument('--finetuned_adapter_path', required=False, default=None, type=str, help="The path of the pretrained adapter (Adapter-tuning)")
    parser.add_argument('--finetuned_adapterlist_path', required=False, default=None, type=str, help="The path of the pretrained adapters list (AdapterFusion)")
    parser.add_argument('--finetuned_adapterfusion_path', required=False, default=None, type=str, help="The path of the adapterfusion component (AdapterFusion)")
 
    args = parser.parse_args()


    set_seeds(3047)  # torch.manual_seed(3407) is all you need


    # test device
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'


    # creat model saving path
    create_directory(args.save_dir)


    # load pre-trained Tokenizer 
    Tokenizer = BertTokenizer.from_pretrained(args.pretrain_tokenizer_path)


    # load sequence data
    if args.inputfile_types == 'DataMatrix': 
        # use to evaluate the model on small datasets
        # do not require any preprocess; one line (raw) includes two columns: first column -> sequence, second column -> label 
        data = pd.read_table(args.data_path, header=None).values
        dataset = TRAFICA_Dataset(data)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=Data_Collection)
    elif args.inputfile_types == 'OneLine': 
        # preprocessed data files for large-scale data (training and evaluation)
        # this way can avoid hardware memory overflow
        dataloader, dataset = DataLoader_with_SeqFolder(args.data_path, batch_size=args.batch_size)
    else:
        raise TypeError('MakePrediction: The format of input data files is not support; Pls make sure inputfile_types is DataMatrix or OneLine')    
 

    # initial a TRAFICA instance
    model = TRAFICA(PreTrained_ModelPath=args.pretrained_model_path, 
                    FineTuned_FullModelPath=args.finetuned_fullmodel_path, 
                    FineTuned_AdapeterPath=args.finetuned_adapter_path, 
                    FineTuned_AdapetersListPath=args.finetuned_adapterlist_path, 
                    FineTuned_AdapeterFusionPath=args.finetuned_adapterfusion_path,
                    FineTuningType=args.task_type)  
    
    # switch model into evaluation mode; added loading steps over PyTorch model.eval()
    model._eval() # do not call this funtion again


    # GPU acceleration
    if torch.cuda.is_available():
        model.to(device)
    

    # output adapterfusion attention -> generate a .h5 in the result folder containing the attention scores of each adapter with respect to the input sequences
    if args.out_att_adapters: 
        if args.task_type != 'AdapterFusion':
            import sys
            sys.exit('Adapters attention can be obtained only in the AdapterFusion phase')
        test_y, test_y_hat, att_AdapterFusion = test_model_outFusionAtt(model, test_data=data, Tokenizer=Tokenizer, device=device)           
        with h5py.File(os.path.join(args.save_dir, 'att_adapters.h5') , 'w') as file:
            for i in range(len(model.BertModel.encoder.layer)):
                file.create_dataset('att_layer_'+str(i), data=att_AdapterFusion[i].numpy())
    else:
        test_y, test_y_hat = test_model(model, test_data=dataloader, Tokenizer=Tokenizer, device=device)


    # save the predicted result
    save_txt_single_column(os.path.join(args.save_dir, 'test_y.txt'), test_y)
    save_txt_single_column(os.path.join(args.save_dir, 'test_y_hat.txt'), test_y_hat)


    # calculate and save the evaluation results
    if args.evaluation_metric == 'set1':
        auroc = metrics.roc_auc_score(y_true=test_y, y_score=test_y_hat)
        save_txt_single_column(os.path.join(args.save_dir, 'auroc.txt'), auroc)
    elif args.evaluation_metric == 'set2':
        r2 = metrics.r2_score(test_y, test_y_hat)
        pcc, _ = pearsonr(test_y, test_y_hat)
        save_txt_single_column(os.path.join(args.save_dir, 'r2.txt'), r2)
        save_txt_single_column(os.path.join(args.save_dir, 'pcc.txt'), pcc)



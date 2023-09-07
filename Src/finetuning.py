from transformers import BertConfig, BertTokenizer
from SeqFolder import SeqFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import sys
import sklearn.metrics as metrics
from scipy.stats import pearsonr
import argparse
import json
from models import Bert_seqClassification, Bert_seqRegression, Bert_seqRegression_adapter, Bert_seqClassification_adapter, Bert_seqRegression_adapterfusion
import faulthandler


sys.path.append('..')
from utils import *



def data_collate_fn(arr):   # padding with [PAD] / tokenizing bases into numbers when loading sequence data
    collect_x = []
    collect_y = []
    for k_mer in arr:
        tmp =k_mer[:-1]
        y = k_mer[-1]
        collect_x.append(' '.join(tmp))
        collect_y.append(float(y))
        #collect_y.append(int(y))
    return collect_x, collect_y
    #return Tokenizer(collect, return_tensors="pt", padding=True), collect_y




def load_datasets(data_path, batch_size, k=4, train_num=None, val_num=None):

    train_data_path = os.path.join(data_path, 'train.'+ str(k) +'mer')
    val_data_path = os.path.join(data_path, 'val.'+ str(k) +'mer')
    test_data_path = os.path.join(data_path, 'test.'+ str(k) +'mer')

    train_data = SeqFolder(train_data_path, max_sample=train_num)
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=data_collate_fn)

    val_data = SeqFolder(val_data_path, max_sample=val_num)
    if val_data.__len__() < batch_size:
        val_batch_size = val_data.__len__()
    else:
        val_batch_size = batch_size
    val_dataloader = DataLoader(val_data, val_batch_size, shuffle=True, collate_fn=data_collate_fn)
    test_data = SeqFolder(test_data_path)

    test_dataloader = DataLoader(test_data, batch_size, shuffle=False, collate_fn=data_collate_fn)

    print('train sample size:\t', train_data.__len__())
    print('val sample size:\t', val_data.__len__())
    print('test sample size:\t', test_data.__len__())

    return train_dataloader, val_dataloader, test_dataloader



def compute_metrics(label, score, evaluation_metric='auroc'):
    if evaluation_metric == 'auroc':
        performance = metrics.roc_auc_score(y_true=label, y_score=score)
    elif evaluation_metric == 'auprc':
        performance = metrics.average_precision_score(y_true=label, y_score=score)
    elif evaluation_metric == 'r2':
        performance = metrics.r2_score(label, score)
    elif evaluation_metric == 'pcc':
        performance, _ = pearsonr(label, score)
    elif evaluation_metric == 'mse':
        performance = metrics.mean_absolute_error(label, score)
    else:
        performance = None
    return performance 



def val_model(model, val_data, Tokenizer, criterion):
    model.eval()
    collect_loss = []
    for data in val_data:
        X, y = data
        inputs = Tokenizer(X, return_tensors="pt", padding=True)
        y = torch.tensor(data[1]).float()
        out = model(inputs.to(device))
        val_loss = criterion(torch.flatten(out), y.to(device))
        # loss
        collect_loss.append(val_loss.data.cpu().detach().numpy())  
    return np.mean(collect_loss)



def train_ft(model,
        train_dataloader,
        val_dataloader, 
        epoches,
        lr,
        Tokenizer,
        model_save_dir,
        earlystopping_tolerance,
        loss_type,
        training_info):

    if loss_type == 'BCE':
        criterion = nn.BCELoss(reduction='mean')
    elif loss_type == 'MSE':
        criterion = nn.MSELoss(reduction='mean')

    optim = torch.optim.AdamW(model.parameters(), lr) # , betas=(0.9, 0.98), eps=1e-07)

    train_loss_epoch =[]
    step_loss_collect = []
    val_loss_epoch = []
    val_loss_best = 999999
    earlystopping_watchdog = 0

    for epoch in range(epoches):
        model.train()
        for data in train_dataloader:
            optim.zero_grad()

            X, y = data
            inputs = Tokenizer(X, return_tensors="pt", padding=True)
            y = torch.tensor(data[1]).float()

            out = model(inputs.to(device))
            loss = criterion(torch.flatten(out), y.to(device))
            loss.backward()
            optim.step()

            step_loss_collect.append(loss.data.cpu().detach().numpy())
            
        train_loss_epoch.append(np.mean(np.array(step_loss_collect)))

        # model validation in each epoch 
        val_loss = val_model(model, val_dataloader, Tokenizer, criterion)
        val_loss_epoch.append(val_loss)
        print('epoch:%d, train loss:%.5f, val loss:%.5f' % (epoch+1, train_loss_epoch[-1], val_loss_epoch[-1]))

        # test_y, test_y_hat = test_model(model, test_data=val_dataloader, Tokenizer=Tokenizer, device=device)
        # metric_score = compute_metrics(test_y, test_y_hat, evaluation_metric='pcc')
        # metric_score1 = compute_metrics(test_y, test_y_hat, evaluation_metric='r2')
        # print(metric_score, metric_score1)

        # save best model
        if val_loss < val_loss_best:
            earlystopping_watchdog = 0
            val_loss_best = val_loss
            model.save_finetuned(model_save_dir)
        
                 
        # early stopping
        earlystopping_watchdog+=1
        if earlystopping_watchdog > earlystopping_tolerance:
            save_txt_single_column(os.path.join(model_save_dir, 'train_loss_epoch.txt'), train_loss_epoch)
            save_txt_single_column(os.path.join(model_save_dir, 'val_loss_epoch.txt'), val_loss_epoch)      
            with open(os.path.join(model_save_dir, 'training_setting.txt'), 'w') as json_file:
                json.dump(training_info, json_file)                       
            return True # stop training procedure



def test_model(model, test_data, Tokenizer, device):
    model.eval()
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

    
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    # training params setting 
    parser.add_argument('--batch_size', required=True, type=int, help="Batch size.")
    parser.add_argument('--n_epoches', required=False, default=0, type=int, help="The number of total training steps/batches.")
    parser.add_argument('--n_epoches_tolerance', required=False, default=0, type=int, help="The number of total training steps/batches.")
    parser.add_argument('--lr', required=True, type=float, help="Maximum learning rate.")
    parser.add_argument('--max_len_tokens', required=False, default=512, type=int, help="Maximum number of tokens in a sample.")
    parser.add_argument('--k', required=True, default=4, type=int, help="K-mer token hyper-parameter.")

    # path
    parser.add_argument('--save_dir', required=True,type=str, help="The saving path of the trained model")
    parser.add_argument('--save_dir_metric', required=True,type=str, help="The saving path of the metric record files")
    parser.add_argument('--data_path', required=True, type=str, help="The path of the data folder")
    parser.add_argument('--vocab_path', required=True, type=str, help="The path of the vocabulary")
    parser.add_argument('--pretrained_model_path', required=True, type=str, help="The path of the pretrained model")
    parser.add_argument('--name_experiment', required=True, default='Example_name', type=str, help="The identifier of this finetuning experiment")
    parser.add_argument('--pretrained_adapters_path', required=False, type=str, help="The path of the pretrained adapters")

    # tool setting
    parser.add_argument('--use_gpu', default=False, type=bool, help="Turn on to use gpu to train model; False: using cpu, True: using gpu if available.")

    # others
    parser.add_argument('--task_type', default='classification', type=str, help="classification or regression.")
    parser.add_argument('--pool_strategy', default='t_cls', type=str, help="Use [CLS] embedding or average all k-mer embeddings as sequence-level representation.")
    parser.add_argument('--max_num_trainsample', default=None, type=int, help="The number of sample used to train the model.")
    parser.add_argument('--max_num_valsample', default=None, type=int, help="The number of sample used to validate the model during training procedure.")

    args = parser.parse_args()

    set_seeds(3047)  # torch.manual_seed(3407) is all you need

    # training device
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

       
    # load pre-trained model & tokenizer initialization
    Tokenizer = BertTokenizer(args.vocab_path, do_lower_case=False, model_max_length=args.max_len_tokens)


    training_info = {'batch_size': args.batch_size,           
                    'lr': args.lr,
                    'k': args.k,
                    'n_epoch': args.n_epoches,
                    'n_epoch_tolerancech_': args.n_epoches_tolerance,
                    'max_len_tokens': args.max_len_tokens,
                    'model_save_dir': args.save_dir,
                    'data_path': args.data_path,
                    'vocab_path': args.vocab_path,
                    'pretrained_model_path': args.pretrained_model_path,
                    'use_gpu': args.use_gpu
                    }

    print(training_info)

    # create saving folders
    create_directory(args.save_dir_metric)
    create_directory(args.save_dir)


    # datasets of each experiment
    train, val, test = load_datasets(args.data_path, batch_size=args.batch_size, k=args.k, train_num=args.max_num_trainsample, val_num=args.max_num_valsample)
    
    # load pre-trained and construct fine-tune model 
    configuration = BertConfig.from_pretrained(args.pretrained_model_path)
     
    if args.task_type == 'classification':
        f_loss = 'BCE'
        evaluation_metric = ['auroc', 'auprc']
        model = Bert_seqClassification(configuration, args.pretrained_model_path, pool_strategy=args.pool_strategy)  
    elif args.task_type == 'regression': 
        f_loss = 'MSE'
        evaluation_metric = ['r2', 'pcc', 'mse']
        model = Bert_seqRegression(configuration, args.pretrained_model_path, pool_strategy=args.pool_strategy)   
    elif args.task_type == 'regression_adapter': 
        f_loss = 'MSE'
        evaluation_metric = ['r2', 'pcc', 'mse']
        model = Bert_seqRegression_adapter(pretrain_modelpath=args.pretrained_model_path, model_path=args.save_dir, pool_strategy=args.pool_strategy)   
    elif args.task_type == 'classification_adapter':    
        f_loss = 'BCE'
        evaluation_metric = ['auroc', 'auprc']
        model = Bert_seqClassification_adapter(pretrain_modelpath=args.pretrained_model_path, model_path=args.save_dir, pool_strategy=args.pool_strategy)   
    elif args.task_type == 'regression_adapterfusion': 
        f_loss = 'MSE'
        evaluation_metric = ['r2', 'pcc', 'mse']
        with open(args.pretrained_adapters_path, "r") as f:
            adapters = json.load(f)
        model = Bert_seqRegression_adapterfusion(pretrain_modelpath=args.pretrained_model_path, pretrain_adapterpath=adapters, model_path=args.save_dir, pool_strategy=args.pool_strategy)   
        
    else:
        print('error in task_type.')
        sys.exit()


    if torch.cuda.is_available():
        model.to(device)

    print('num of model params:\t',sum([x.nelement() for x in model.parameters()]))

    faulthandler.enable()

    # train with earlystopping
    train_ft(model, train, val, epoches= args.n_epoches, lr=args.lr, Tokenizer=Tokenizer, model_save_dir=args.save_dir, earlystopping_tolerance=args.n_epoches_tolerance, loss_type=f_loss, training_info=training_info)     
    
    del model
    torch.cuda.empty_cache()

    # test
    if args.task_type == 'classification':
        model_used2test = Bert_seqClassification(configuration, args.save_dir, pool_strategy=args.pool_strategy, fine_tuned=True)
    elif args.task_type == 'regression':
        model_used2test = Bert_seqRegression(configuration, args.save_dir, pool_strategy=args.pool_strategy, fine_tuned=True)
    elif args.task_type == 'regression_adapter': 
        model_used2test = Bert_seqRegression_adapter(pretrain_modelpath=args.pretrained_model_path, model_path=args.save_dir, pool_strategy=args.pool_strategy, fine_tuned=True)   
    elif args.task_type == 'classification_adapter': 
        model_used2test = Bert_seqClassification_adapter(pretrain_modelpath=args.pretrained_model_path, model_path=args.save_dir, pool_strategy=args.pool_strategy, fine_tuned=True)   
    elif args.task_type == 'regression_adapterfusion': 
        model_used2test = Bert_seqRegression_adapterfusion(pretrain_modelpath=args.pretrained_model_path, pretrain_adapterpath=adapters, model_path=args.save_dir, pool_strategy=args.pool_strategy, fine_tuned=True)   
 
    # print(model_used2test)
    # debug_breakpoint()

    if torch.cuda.is_available():
        model_used2test.to(device)

    test_y, test_y_hat = test_model(model_used2test, test_data=test, Tokenizer=Tokenizer, device=device)
    
    # saving test result
    save_txt_single_column(os.path.join(args.save_dir, 'test_y.txt'), test_y)
    save_txt_single_column(os.path.join(args.save_dir, 'test_y_hat.txt'), test_y_hat)

    for metric in evaluation_metric:
        metric_score = compute_metrics(test_y, test_y_hat, evaluation_metric=metric)
        saving_test_performance(args.save_dir_metric, metric_score, 'test_' + metric, args.name_experiment) # saving evaluation results 


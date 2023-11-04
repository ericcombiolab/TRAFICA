# transformer and PyTorch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

# others
import numpy as np
import sys
import sklearn.metrics as metrics
from scipy.stats import pearsonr
import argparse
import json
import faulthandler

# TRAFICA
from models import TRAFICA
from dataset import DataLoader_with_SeqFolder, Data_Collection, TRAFICA_Dataset

sys.path.append('..')
from utils import *



def LoadDataset_with_SeqFolder(train_path, val_path=None, test_path=None, batch_size=128, 
                train_num=None, val_num=None, split_ratio=[0.8, 0.1], size_print=False):

    # fine-tuning the model with evaluation
    if val_path and test_path:
        train_dataloader, train_data = DataLoader_with_SeqFolder(train_path, max_sample=train_num, shuffle=True, batch_size=batch_size)
        val_dataloader, val_data = DataLoader_with_SeqFolder(val_path, max_sample=val_num, shuffle=True, batch_size=batch_size) 
        test_dataloader, test_data = DataLoader_with_SeqFolder(test_path, shuffle=False, batch_size=batch_size)

        # print the sizes of three sets
        if size_print:
            print('training set size:\t', train_data.__len__())
            print('validation set size:\t', val_data.__len__())
            print('test set size:\t', test_data.__len__())

        return train_dataloader, train_data, val_dataloader, val_data, test_dataloader, test_data
    
    # fine-tuning the model without evaluation
    elif val_path:
        train_dataloader, train_data = DataLoader_with_SeqFolder(train_path, max_sample=train_num, shuffle=True, batch_size=batch_size)
        val_dataloader, val_data = DataLoader_with_SeqFolder(val_path, max_sample=val_num, shuffle=True, batch_size=batch_size) 
        test_dataloader = None
        test_data = None
        # print the sizes of three sets
        if size_print:
            print('training set size:\t', train_dataloader.__len__())
            print('validation set size:\t', val_dataloader.__len__())
            print('Not test set; fine-tuning the model without evaluation')

        return train_dataloader, train_data, val_dataloader, val_data, test_dataloader, test_data
    
    # TODO: split sets from a input dataset automatically by the input ratio (split_ratio)
    else:
        raise NotImplementedError('Dateset automatically split (TODO further): Need to munaully split datasets (training, validation, and test) and specify their path in current.')



def LoadDataset(train_path, val_path=None, test_path=None, batch_size=128, 
                               train_num=None, val_num=None, split_ratio=[0.8, 0.1], size_print=False):
    
    def _load_from_datamatrix(path, max_sample=None, shuffle=False, batch_size=128):
        data = pd.read_table(path, header=None).values
        # sample a subset
        if max_sample:
            if max_sample < len(data):
                # top sequences ranked by affinities
                sorted_indices = np.argsort(data[:, 1])[::-1]
                data = data[sorted_indices]
                bin_idx= quantile_bins(data, max_sample)
                subset = sample_from_bins(data, bin_idx)
                data = subset
                # # random
                # num_rows = data.shape[0]
                # random_indices = np.random.choice(num_rows, max_sample, replace=False)
                # data = data[random_indices]
                # #
                # # data = data[:max_sample,:]

        dataset = TRAFICA_Dataset(data)
        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=Data_Collection)
        return dataloader, dataset
        
    # fine-tuning the model with evaluation
    if val_path and test_path:

        train_dataloader, train_data = _load_from_datamatrix(train_path, max_sample=train_num, shuffle=True, batch_size=batch_size)
        val_dataloader, val_data = _load_from_datamatrix(val_path, max_sample=val_num, shuffle=True, batch_size=batch_size) 
        test_dataloader, test_data = _load_from_datamatrix(test_path, shuffle=False, batch_size=batch_size)

        # print the sizes of three sets
        if size_print:
            print('training set size:\t', train_data.__len__())
            print('validation set size:\t', val_data.__len__())
            print('test set size:\t', test_data.__len__())

        return train_dataloader, train_data, val_dataloader, val_data, test_dataloader, test_data
    
    # fine-tuning the model without evaluation
    elif val_path:
        train_dataloader, train_data = _load_from_datamatrix(train_path, max_sample=train_num, shuffle=True, batch_size=batch_size)
        val_dataloader, val_data = _load_from_datamatrix(val_path, max_sample=val_num, shuffle=True, batch_size=batch_size) 
        test_dataloader = None
        test_data = None
        # print the sizes of three sets
        if size_print:
            print('training set size:\t', train_dataloader.__len__())
            print('validation set size:\t', val_dataloader.__len__())
            print('Not test set; fine-tuning the model without evaluation')

        return train_dataloader, train_data, val_dataloader, val_data, test_dataloader, test_data
    
    # TODO: split sets from a input dataset automatically by the input ratio (split_ratio)
    else:
        raise NotImplementedError('Dateset automatically split (TODO further): Need to munaully split datasets (training, validation, and test) and specify their path in current.')


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
    model.eval() # switch the model to evaluation mode
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



def fine_tuning(model,
        train_dataloader,
        val_dataloader, 
        epoches,
        earlystopping_tolerance,
        lr,
        Tokenizer,
        model_save_dir,
        training_info):

    # loss function
    if model.PredictorType == 'classification':
        criterion = nn.BCELoss(reduction='mean')
    elif model.PredictorType == 'regression':
        criterion = nn.MSELoss(reduction='mean')

    # optimizer
    optim = torch.optim.AdamW(model.parameters(), lr) # , betas=(0.9, 0.98), eps=1e-07)

    # loss recored and earlystopping
    train_loss_epoch =[]
    step_loss_collect = []
    val_loss_epoch = []
    val_loss_best = 999999
    earlystopping_watchdog = 0

    # fine-tuning
    for epoch in range(epoches):
        model.train() # switch the model to training mode
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
            

            # break # debug


        train_loss_epoch.append(np.mean(np.array(step_loss_collect)))

        # model validation in each epoch 
        val_loss = val_model(model, val_dataloader, Tokenizer, criterion)
        val_loss_epoch.append(val_loss)
        print('epoch:%d, train loss:%.5f, val loss:%.5f' % (epoch+1, train_loss_epoch[-1], val_loss_epoch[-1]))

        # save best model
        if val_loss < val_loss_best:
            earlystopping_watchdog = 0
            val_loss_best = val_loss
            model.prediction_model.save_finetuned(model_save_dir)
                 
        # early stopping monitor in each epoch
        earlystopping_watchdog+=1
        if earlystopping_watchdog > earlystopping_tolerance:
            save_txt_single_column(os.path.join(model_save_dir, 'train_loss_epoch.txt'), train_loss_epoch)
            save_txt_single_column(os.path.join(model_save_dir, 'val_loss_epoch.txt'), val_loss_epoch)      
            with open(os.path.join(model_save_dir, 'training_setting.txt'), 'w') as json_file:
                json.dump(training_info, json_file)                       
            return True # stop fine-tuning procedure

        # break # debug


def test_model(model, test_dataloader, Tokenizer, device):
    collect_y = []
    collect_out = []
    for data in test_dataloader:
        X, y = data
        inputs = Tokenizer(X, return_tensors="pt", padding=True)
        y = torch.tensor(data[1]).float()
      
        out = model(inputs.to(device))

        collect_y += y.tolist()
        collect_out += out.flatten().cpu().detach().numpy().tolist()

        # break # debug

    return  collect_y, collect_out


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    # fine-tuning setting 
    parser.add_argument('--batch_size', required=True, type=int, help="Batch size.")
    parser.add_argument('--n_epoches', required=False, default=0, type=int, help="The number of total training steps/batches.")
    parser.add_argument('--n_epoches_tolerance', required=False, default=0, type=int, help="The number of total training steps/batches.")
    parser.add_argument('--lr', required=True, type=float, help="Maximum learning rate.")
    parser.add_argument('--use_gpu', default=False, type=bool, help="Turn on to use gpu to train model; False: using cpu, True: using gpu if available.")
    parser.add_argument('--max_num_train', default=None, type=int, help="The number of samples used to train the model in fine-tuning.")
    parser.add_argument('--max_num_val', default=None, type=int, help="The number of samples used to validate the model in fine-tuning.")

    # basic setting
    parser.add_argument('--inputfile_types', default='DataMatrix', type=str, help="OneLine file or DataMatrix")
    parser.add_argument('--task_type', default='AdapterTuning', type=str, help="TRAFICA support three types of fine-tuning: FullyFineTuning, AdapterTuning, and AdapterFusion")
    parser.add_argument('--predict_type', default='regression', type=str, help="TRAFICA support two types of prediction: regression and classification")
    parser.add_argument('--name_experiment', required=False, default='Example_name', type=str, help="The identifier of the finetuning model")

    # path
    parser.add_argument('--save_dir_metric', required=False,type=str, help="The saving path of the metric record files")
    parser.add_argument('--save_dir', required=True,type=str, help="The saving path of the finetuned model and/or evaluation result")
    parser.add_argument('--train', required=True, type=str, help="The path of the training set file/folder")
    parser.add_argument('--val', required=False, type=str, help="The path of the validation set file/folder")
    parser.add_argument('--test', required=False, type=str, help="The path of the test set file/folder")
    parser.add_argument('--pretrain_tokenizer_path', required=True, type=str, help="The path of the tokenizer (4-mer)")
    parser.add_argument('--pretrained_model_path', required=False, default=None, type=str, help="The path of the pretrained model")
    parser.add_argument('--finetuned_fullmodel_path', required=False, default=None, type=str, help="The path of the fully finetuned model (Fully fine-tuning)")
    parser.add_argument('--finetuned_adapter_path', required=False, default=None, type=str, help="The path of the pretrained adapter (Adapter-tuning)")
    parser.add_argument('--finetuned_adapterlist_path', required=False, default=None, type=str, help="The path of the pretrained adapters list (AdapterFusion)")
    parser.add_argument('--finetuned_adapterfusion_path', required=False, default=None, type=str, help="The path of the adapterfusion component (AdapterFusion)")
 

    args = parser.parse_args()

    set_seeds(3047)  # torch.manual_seed(3407) is all you need

    # training device
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'


    # # create saving folders
    create_directory(args.save_dir_metric)
    create_directory(args.save_dir)


    # load pre-trained Tokenizer 
    Tokenizer = BertTokenizer.from_pretrained(args.pretrain_tokenizer_path)


    # load sequence data
    if args.inputfile_types == 'DataMatrix': 
        # use to evaluate the model on small datasets
        # do not require any preprocess; one line (raw) includes two columns: first column -> sequence, second column -> label 
        train_loader, train_data, val_loader, val_data, test_loader, test_data = LoadDataset(train_path=args.train, 
                                                                                            val_path=args.val, 
                                                                                            test_path=args.test, 
                                                                                            batch_size=args.batch_size, 
                                                                                            train_num=args.max_num_train, 
                                                                                            val_num=args.max_num_val)
    elif args.inputfile_types == 'OneLine': 
        # preprocessed data files for large-scale data (training and evaluation)
        # this way can avoid hardware memory overflow
        train_loader, train_data, val_loader, val_data, test_loader, test_data = LoadDataset_with_SeqFolder(train_path=args.train, 
                                                                                                        val_path=args.val, 
                                                                                                        test_path=args.test, 
                                                                                                        batch_size=args.batch_size, 
                                                                                                        train_num=args.max_num_train, 
                                                                                                        val_num=args.max_num_val)
    else:
        raise TypeError('MakePrediction: The format of input data files is not support; Pls make sure inputfile_types is DataMatrix or OneLine')    
 

    # initial a TRAFICA instance
    model = TRAFICA(PreTrained_ModelPath=args.pretrained_model_path, 
                    FineTuned_FullModelPath=args.finetuned_fullmodel_path, 
                    FineTuned_AdapeterPath=args.finetuned_adapter_path, 
                    FineTuned_AdapetersListPath=args.finetuned_adapterlist_path, 
                    FineTuned_AdapeterFusionPath=args.finetuned_adapterfusion_path,
                    FineTuningType=args.task_type,
                    PredictorType=args.predict_type) 
    
    # switch model into training mode; added loading steps over PyTorch model.train()
    model._train() # do not call this funtion again


    # GPU acceleration
    if torch.cuda.is_available():
        model.to(device)


    # print the size of the model
    print('num of model params:\t',sum([x.nelement() for x in model.parameters()]))


    # handle fault
    faulthandler.enable()


    # basic fine-tuning information
    training_info = {'batch_size': args.batch_size,           
                    'lr': args.lr,
                    'n_epoch': args.n_epoches,
                    'n_epoch_tolerancech_': args.n_epoches_tolerance,
                    'task_type': args.task_type,
                    'predict_type': args.predict_type, 
                    'trainset_size': train_data.__len__(), 
                    'valset_size': val_data.__len__(),
                    'n_modelparams': sum([x.nelement() for x in model.parameters()])
                    }
    print(training_info)
 

    # fine-tune the model with earlystopping
    fine_tuning(model, 
                train_loader, 
                val_loader, 
                epoches= args.n_epoches,
                earlystopping_tolerance=args.n_epoches_tolerance,  
                lr=args.lr, 
                Tokenizer=Tokenizer, 
                model_save_dir=args.save_dir,  
                training_info=training_info)     
    
    # delete the cache of the fine-tuning model
    del model
    torch.cuda.empty_cache()



    # evaluation if input test data
    if test_loader:
        # initial a TRAFICA instance
        model_used2test = TRAFICA(PreTrained_ModelPath=args.pretrained_model_path, 
                        FineTuned_FullModelPath=args.save_dir, 
                        FineTuned_AdapeterPath=args.save_dir, 
                        FineTuned_AdapetersListPath=args.finetuned_adapterlist_path, 
                        FineTuned_AdapeterFusionPath=args.save_dir,
                        FineTuningType=args.task_type,
                        PredictorType=args.predict_type) 
        
        # switch model into training mode; added loading steps over PyTorch model.train()
        model_used2test._eval() # do not call this funtion again

        # GPU acceleration
        if torch.cuda.is_available():
            model_used2test.to(device)


        # prediction 
        test_y, test_y_hat = test_model(model_used2test, test_dataloader=test_loader, Tokenizer=Tokenizer, device=device)
        

        # saving predicted results
        save_txt_single_column(os.path.join(args.save_dir, 'test_y.txt'), test_y)
        save_txt_single_column(os.path.join(args.save_dir, 'test_y_hat.txt'), test_y_hat)


        # evaluation
        if model_used2test.PredictorType == 'regression':
            evaluation_metric = ['r2', 'pcc', 'mse']
        elif model_used2test.PredictorType == 'classification':
            evaluation_metric = ['auroc', 'auprc']
        else:
            raise TypeError('Evaluation error: TRAFICA support two types of prediction: regression and classification')

        for metric in evaluation_metric:
            metric_score = compute_metrics(test_y, test_y_hat, evaluation_metric=metric)    
            save_txt_single_column(os.path.join(args.save_dir, metric+'.txt'), metric_score)
     
            # for batch fine-tuning to save evaluation results into one .json file
            if args.save_dir_metric and args.name_experiment:
                saving_test_performance(args.save_dir_metric, metric_score, 'test_' + metric, args.name_experiment) # saving evaluation results 


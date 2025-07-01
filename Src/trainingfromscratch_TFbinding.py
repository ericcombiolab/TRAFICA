
from transformers import RoFormerConfig, AutoTokenizer, RoFormerForSequenceClassification 
import torch.nn as nn
import torch
import numpy as np
import sklearn.metrics as metrics
from scipy.stats import pearsonr
import argparse
import json
import faulthandler
from TRAFICA.dataset import load_dataset_from_txt
from tqdm import tqdm 
from util import *




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



def validation(model, val_dataloader, Tokenizer, criterion, tokenization,sp):
    model.eval() # switch the model to evaluation mode
    collect_loss = []
    # for data in val_dataloader:
    for data in tqdm(val_dataloader, desc='Validating TF-binding',total=len(val_dataloader)):
        
        sequences, y = data
        if tokenization == 'BPE_DNABERT':
            inputs = Tokenizer(sequences, return_tensors='pt', padding=True).to(device)
        else:
            tokens_batch = piece_sequences(sequences, tokenization, sp)   
            inputs = Tokenizer(tokens_batch, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            
        logit = outputs.logits     
        y = torch.tensor(y).float()
        val_loss = criterion(torch.flatten(logit), y.to(device))
        collect_loss.append(val_loss.data.cpu().detach().numpy())  
        
    return np.mean(collect_loss)




def testing(model, test_dataloader, Tokenizer, tokenization, sp, device):
    model.eval() # switch the model to evaluation mode
    collect_y = []
    collect_out = []
    for data in tqdm(test_dataloader, desc='Testing TF-binding',total=len(test_dataloader)):
        
        sequences, y = data
        if tokenization == 'BPE_DNABERT':
            inputs = Tokenizer(sequences, return_tensors='pt', padding=True).to(device)
        else:
            tokens_batch = piece_sequences(sequences, tokenization, sp)   
            inputs = Tokenizer(tokens_batch, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            
        logit = outputs.logits  
           
        collect_y += y.tolist()
        collect_out += logit.flatten().cpu().detach().numpy().tolist()

        # break # debug

    return  collect_y, collect_out


def fine_tuning(model,
        train_dataloader,
        val_dataloader, 
        n_epoch,
        n_toler,
        lr,
        Tokenizer,
        save_dir,
        training_info,
        predict_type,
        tokenization,
        sp):
    

    # loss function
    if predict_type == 'classification':
        criterion = nn.BCELoss(reduction='mean')
    elif predict_type == 'regression':
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
    for epoch in range(n_epoch):

        model.train() # switch the model to training mode
        for data in tqdm(train_dataloader, desc='Fine-tuning TF-binding',total=len(train_dataloader)):
            optim.zero_grad()
            
            sequences, y = data
            if tokenization == 'BPE_DNABERT':
                inputs = Tokenizer(sequences, return_tensors='pt', padding=True).to(device)
            else:
                tokens_batch = piece_sequences(sequences, tokenization, sp)   
                inputs = Tokenizer(tokens_batch, return_tensors="pt", padding=True).to(device)

            outputs = model(**inputs)
            logit = outputs.logits
            y = torch.tensor(y).float()
            loss = criterion(torch.flatten(logit), y.to(device))
            loss.backward()
            optim.step()

            step_loss_collect.append(loss.data.cpu().detach().numpy())


            # break # debug


        train_loss_epoch.append(np.mean(np.array(step_loss_collect)))

        # model validation in each epoch 
        val_loss = validation(model, val_dataloader, Tokenizer, criterion,tokenization,sp)
        val_loss_epoch.append(val_loss)
        
     
        # save best model
        if val_loss < val_loss_best:
            earlystopping_watchdog = 0
            val_loss_best = val_loss
            # model.save_pretrained(os.path.join(save_dir,"lora_adapter")) # saving LoRA module weights
            # torch.save({'PREDICT_HEAD' : model.classifier.state_dict()}, os.path.join(save_dir,"predict_head_weights.pth"))
            # torch.save({'Model_Weights': model.base_model.model.state_dict()}, os.path.join(save_dir,"model_weights.pth"))
            save_txt_single_column(train_loss_epoch, save_dir, 'train_loss_epoch.txt')
            save_txt_single_column(val_loss_epoch, save_dir, 'val_loss_epoch.txt')
            
        print('epoch:%d, train loss:%.10f, val loss:%.10f, watchdog:%d' % (epoch+1, train_loss_epoch[-1], val_loss_epoch[-1], earlystopping_watchdog))
                        
        # early stopping monitor in each epoch
        earlystopping_watchdog+=1
        if earlystopping_watchdog > n_toler:  
            save_txt_single_column(train_loss_epoch, save_dir, 'train_loss_epoch.txt')
            save_txt_single_column(val_loss_epoch, save_dir, 'val_loss_epoch.txt')
            
            with open(os.path.join(save_dir, 'training_setting.txt'), 'w') as json_file:
                json.dump(training_info, json_file)                       
            return True # stop fine-tuning procedure

               
        # break #  debug




if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    # fine-tuning setting 
    parser.add_argument('--mode', default='train_from_scratch', type=str, help=".")
    
    parser.add_argument('--batch_size', required=True, type=int, help="Batch size.")
    parser.add_argument('--n_epoch', required=False, default=0, type=int, help="The number of total training steps/batches.")
    parser.add_argument('--n_toler', required=False, default=0, type=int, help="The number of total training steps/batches.")
    parser.add_argument('--lr', required=True, type=float, help="Maximum learning rate.")
    parser.add_argument('--use_gpu', default=False, type=bool, help="Turn on to use gpu to train model; False: using cpu, True: using gpu if available.")

    # basic setting
    parser.add_argument('--predict_type', default='regression', type=str, help=".")
    
    # path
    parser.add_argument('--save_dir', required=True,type=str, help="The saving path of the finetuned model and/or evaluation result")
    parser.add_argument('--eval_data_path', required=True, type=str, help="The path of the training set file/folder")
    parser.add_argument('--tokenizer_path', required=True, type=str, help="The path of the tokenizer (4-mer)")
    parser.add_argument('--pretrained_model_path', required=False, default=None, type=str, help="The path of the pretrained model")
    parser.add_argument('--tokenization', required=False, default='4_mer', type=str, help=".")
    

    args = parser.parse_args()

    set_seeds(3047)  # torch.manual_seed(3407) is all you need



    # create saving folder
    create_directory(args.save_dir)

    # training device
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'
        
        
    # data    

    train_loader= load_dataset_from_txt(os.path.join(args.eval_data_path, 'train.txt'), 
                                        shuffle=True,
                                        batch_size=args.batch_size)
        
    val_loader= load_dataset_from_txt(os.path.join(args.eval_data_path, 'val.txt'), 
                                        shuffle=True,
                                        batch_size=args.batch_size)    
    
    test_loader= load_dataset_from_txt(os.path.join(args.eval_data_path, 'test.txt'), 
                                        shuffle=False,
                                        batch_size=args.batch_size)


    
    # tokenizer
    if args.tokenization == 'BPE':
        sp = spm.SentencePieceProcessor()
        BPE_path = os.path.join('./Tokenizers','BPE_Processing', 'dna_BPE_model.model')
        sp.load(BPE_path)
    else:
        sp=None
 
    Tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)    
        
        
    # model    
    config = RoFormerConfig.from_pretrained(args.pretrained_model_path)
    config.num_labels = 1 
    if args.mode == 'train_from_scratch':
        model = RoFormerForSequenceClassification(config=config)
    else:
        model = RoFormerForSequenceClassification.from_pretrained(args.pretrained_model_path, config=config)
    
    
    for param in model.parameters():             # open weights
        param.requires_grad = True    
            
    model.to(device)  
    
      
    # print the size of the model
    print('num of model params:\t',sum([x.nelement() for x in model.parameters()]))
    
    # handle fault
    faulthandler.enable()

    # basic fine-tuning information
    training_info = {'batch_size': args.batch_size,           
                    'lr': args.lr,
                    'n_epoch': args.n_epoch,
                    'n_toler': args.n_toler,
                    'predict_type': args.predict_type, 
                    'trainset_size': len(train_loader.dataset), 
                    'valset_size': len(val_loader.dataset),
                    'n_params': sum([x.nelement() for x in model.parameters()])
                    }
    print(training_info)
    
    
    # fine-tune the model with earlystopping
    fine_tuning(model, 
                train_loader, 
                val_loader, 
                n_epoch= args.n_epoch,
                n_toler=args.n_toler,  
                lr=args.lr, 
                Tokenizer=Tokenizer, 
                save_dir=args.save_dir,  
                training_info=training_info,
                predict_type=args.predict_type,
                tokenization=args.tokenization,
                sp=sp)    

    torch.cuda.empty_cache()
    

    model.eval()
    test_y, test_y_hat = testing(model, test_dataloader=test_loader, Tokenizer=Tokenizer, tokenization=args.tokenization, sp=sp, device=device)
    
    save_txt_single_column(test_y, args.save_dir, 'test_y.txt')
    save_txt_single_column(test_y_hat, args.save_dir, 'test_y_hat.txt')
    
    if args.predict_type == 'regression':
        evaluation_metric = ['r2', 'pcc', 'mse']
    elif args.predict_type == 'classification':
        evaluation_metric = ['auroc', 'auprc']
    else:
        raise TypeError('Evaluation error: TRAFICA support two types of prediction: regression and classification')

    for metric in evaluation_metric:
        metric_score = compute_metrics(test_y, test_y_hat, evaluation_metric=metric)    
        save_txt_single_column([metric_score], args.save_dir, f"{metric}.txt")



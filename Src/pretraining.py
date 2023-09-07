from transformers import BertConfig, BertTokenizer
from models import Bert_MLM_model
from SeqFolder import SeqFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import sklearn.metrics as metrics
import math
import argparse
import wandb
import json
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

sys.path.append('..')
from utils import *



def data_collate_fn(arr): # avoid error
    return arr



def compute_metrics(out, batch_data, mask_idx):
    pred = torch.argmax(F.softmax(out, dim=-1), dim=2).cpu().detach().numpy()  # token index
    label = batch_data.clone().cpu().detach().numpy()
    idx = mask_idx.clone().cpu().detach().numpy()
    target_word_idxs = label[idx]
    predict_word_idxs = pred[idx]  

    accuracy = metrics.accuracy_score(y_true=target_word_idxs, y_pred=predict_word_idxs)
    return accuracy    


def train(model,
        train_dataloader,
        train_data, 
        total_training_step,
        lr,
        lr_warmup,
        batch_size,
        Tokenizer,
        model_save_dir,
        mask_ratio,
        mask_n_phrases,
        training_info):

    ## training steps and epochs
    n_step_epoch = math.ceil( train_data.__len__() / batch_size )
    epochs = int(total_training_step/n_step_epoch) + 1 # based on training steps

    ## suit up for training
    criterion = nn.CrossEntropyLoss(reduction='none')
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-07)
    scheduler = WarmupLR_Linear(optimizer=optim, max_lr=[lr], num_warm=lr_warmup, num_allsteps=total_training_step)
    scaler = GradScaler()  
    model.train()

    ## info record
    currentt_step = 0
    loss_collect = []
    accuracy_collect = []

    for epoch in range(epochs):
        for batch_data in train_dataloader:
            currentt_step +=1
            optim.zero_grad()

            # Mask k-mers 
            if mask_n_phrases > 0: 
                Masked_data, NoMasked_data = mask_strategy_contiguous_split_for_HuggingBert(batch_data, ratio=mask_ratio, n_phrase_mask=mask_n_phrases) 
            else:
                Masked_data, NoMasked_data = mask_strategy_contiguous_for_HuggingBert(batch_data, ratio=mask_ratio) 

            input_data = Tokenizer(Masked_data, return_tensors="pt", padding=True)
            label = Tokenizer(NoMasked_data, return_tensors="pt", padding=True)

            # # add mask to input data
            # input = data['input_ids'].clone().flatten()
            # input[mask_idx] = Tokenizer.get_vocab()['[MASK]']
            # data['input_ids'] = input.view(data['input_ids'].size())
        

            # model forward
            with autocast():
                logit, bert_out = model(input_data.to(device))
                loss = criterion(logit.view(-1, Tokenizer.vocab_size).to(device), label.input_ids.view(-1).to(device))
       
            # only mask loss     
            mask_token_index = (input_data.input_ids == Tokenizer.mask_token_id)
            loss = loss.reshape(mask_token_index.shape)
            loss_masked = loss[mask_token_index]
            # loss_mask = torch.zeros(len(loss))
            # loss_mask[mask_idx] = 1
            # loss_masked = torch.masked_select(loss, loss_mask.bool().to(device)) # calculate loss of [MASK] words       
            loss_masked = torch.mean(loss_masked) 

            scaler.scale(loss_masked).backward()
            scaler.step(optim)
            scaler.update()

            # accuracy 
            acc = compute_metrics(logit, label.input_ids, mask_token_index)
            loss_collect.append(loss_masked.cpu().item())
            accuracy_collect.append(acc)
         
            if args.wandb:
                wandb.log({'learning_rate': optim.param_groups[0]['lr'], 'step_loss': loss_masked, 'step_acc': acc})

            # learning rate warmup & decay
            scheduler.step()

            # prevent the thread to be killed 
            if currentt_step%2000==0:
                print('Finished the num of steps:\t',currentt_step)

            # model saving       
            if currentt_step%20000==0:      
                model.BertModel.save_pretrained(model_save_dir)   
                save_txt_single_column(os.path.join(model_save_dir, 'loss_record.txt') , loss_collect)
                save_txt_single_column(os.path.join(model_save_dir, 'acc_record.txt') , accuracy_collect)
                with open(os.path.join(model_save_dir, 'training_setting.txt'), 'w') as json_file:
                    json.dump(training_info, json_file)
                    
           # terminate training 
            if currentt_step == total_training_step:  
                model.BertModel.save_pretrained(model_save_dir)  
                save_txt_single_column(os.path.join(model_save_dir, 'loss_record.txt') , loss_collect)
                save_txt_single_column(os.path.join(model_save_dir, 'acc_record.txt') , accuracy_collect)
                return True


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # training params setting 
    parser.add_argument('--batch_size', required=True, type=int, help="Batch size, Bert paper default:256.")
    parser.add_argument('--total_training_step', required=False, default=0, type=int, help="The number of total training steps/batches.")
    parser.add_argument('--lr_warmup', required=True, type=int, help="The number of training steps/batches for learning rate warm up.")
    parser.add_argument('--lr', required=True, type=float, help="Maximum learning rate.")
    parser.add_argument('--max_len_tokens', required=False, default=512, type=int, help="Maximum number of tokens in a sample.")
    parser.add_argument('--mask_ratio', required=True, default=0.15, type=float, help="The ratio of masked k-mers in a sequence, Bert paper default: 0.15.")
    parser.add_argument('--mask_n_phrases', required=False, default=2, type=int, help="The number of masked k-mer phrases ( k-mer -> word; continuous k-mers -> phrase).")
  
    # path
    parser.add_argument('--model_save_dir', required=True,type=str, help="The saving path of the trained model")
    parser.add_argument('--train_data_path', required=True, type=str, help="The path of the training data folder")
    parser.add_argument('--vocab_path', required=True, type=str, help="The path of the vocabulary")

    # tool setting
    parser.add_argument('--wandb', default=False, type=bool, help="Turn on the tool for visualization of training progress")
    parser.add_argument('--use_gpu', default=False, type=bool, help="Turn on to use gpu to train model; False: using cpu, True: using gpu if available.")

    args = parser.parse_args()


    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    if args.wandb:
        trace_tool = wandb.init(project="bert_mlm")

    train_data = SeqFolder(args.train_data_path)
    train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True, collate_fn=data_collate_fn)
    Tokenizer = BertTokenizer(args.vocab_path, do_lower_case=False, model_max_length=args.max_len_tokens)
    configuration = BertConfig(vocab_size=Tokenizer.vocab_size, output_attentions=True)
    model = Bert_MLM_model(configuration)


    if torch.cuda.is_available():
        model.to(device)

    # print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    training_info = {'batch_size': args.batch_size,
                    'total_training_step': args.total_training_step,
                    'lr_warmup': args.lr_warmup,
                    'lr': args.lr,
                    'max_len_tokens': args.max_len_tokens,
                    'mask_ratio': args.mask_ratio,
                    'mask_n_phrases': args.mask_n_phrases,
                    'model_save_dir': args.model_save_dir,
                    'train_data_path': args.train_data_path,
                    'vocab_path': args.vocab_path,
                    'wandb': args.wandb,
                    'use_gpu': args.use_gpu,
                    'num_model_params': sum([x.nelement() for x in model.parameters()]),
                    'train_data_size': train_data.__len__()                   
                    }

    print(training_info)


    train(model,
        train_dataloader,
        train_data, 
        args.total_training_step,
        args.lr,
        args.lr_warmup,
        args.batch_size,
        Tokenizer,
        args.model_save_dir,
        args.mask_ratio,
        args.mask_n_phrases,
        training_info)

    if args.wandb:
        trace_tool.finish()


    # example: 
    # python pretrain_BindBERT.py --batch_size 384 --total_training_step 110000 --lr_warmup 10000 --lr 0.0001 --mask_ratio 0.15 --mask_n_phrases 2 --wandb True --use_gpu True --model_save_dir ./pretrained_models/ATAC_seq_2split_4mer --train_data_path /tmp/csyuxu/processed_ATAC_seq_ENCODE_p100_4mer_pretrainDataFolder --vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt
    # python pretrain_BindBERT.py --batch_size 384 --total_training_step 110000 --lr_warmup 10000 --lr 0.0001 --mask_ratio 0.15 --mask_n_phrases 0 --wandb True --use_gpu True --model_save_dir ./pretrained_models/ATAC_seq_nosplit_4mer --train_data_path /tmp/csyuxu/processed_ATAC_seq_ENCODE_p100_4mer_pretrainDataFolder --vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt





from transformers import RoFormerConfig, BertTokenizerFast, AutoTokenizer, RoFormerForMaskedLM  
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
import math
import argparse
import wandb
from torch.amp import autocast
from torch.amp import GradScaler


from TRAFICA.dataset import load_dataset_from_h5
from util import *

from tqdm import tqdm

class WarmupLR:
    def __init__(self, optimizer, max_lr, num_warm, num_allsteps,decay_type='linear') -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        self.lr = max_lr
        self.num_step = 0
        self.num_allsteps = num_allsteps
        self.decay_type = decay_type

    def __compute(self, lr) -> float:
        if self.num_step <= self.num_warm:
            initial_lr = lr*0.1
            return initial_lr + (lr - initial_lr) * (self.num_step / self.num_warm)
        else:   # linear decay
            if self.decay_type == 'linear':
                return lr * (1- ( (self.num_step-self.num_warm) / (self.num_allsteps-self.num_warm) ) )
            elif self.decay_type == 'cosine':
                return lr * 0.5 * (1 + math.cos(math.pi * (self.num_step-self.num_warm) / (self.num_allsteps-self.num_warm) ))

    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr] 
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]

    def get_lr(self):
        return self.lr




def compute_metrics(out, batch_data, mask_idx):
    pred = torch.argmax(F.softmax(out, dim=-1), dim=2).cpu().detach().numpy()  # token index
    label = batch_data.clone().cpu().detach().numpy()
    idx = mask_idx.clone().cpu().detach().numpy()
    target_word_idxs = label[idx]
    predict_word_idxs = pred[idx]  

    accuracy = metrics.accuracy_score(y_true=target_word_idxs, y_pred=predict_word_idxs)
    return accuracy    


def train(model,
        train_loader,
        # val_loader,
        total_steps,
        lr,
        batch_size,
        save_dir,
        mask_ratio:int=0.15,
        tokenization_mode:str='Base-level',
        amp_on: bool=False,
        lr_decay_type:str = 'cosine',
        continuous_mask_tokens:int= 5,
        check_point:int=2000,
        sp=None):

    # model save_dir 
    model_dir = os.path.join(save_dir, 'TRAFICA_Weights')

    # loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=Tokenizer.pad_token_id)
    
    # optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr , betas=(0.9, 0.98), eps=1e-07)   

    # calculating the number of epochs
    n_step_epoch = math.ceil( len(train_loader.dataset) / batch_size )
    n_epoch = int(total_steps/n_step_epoch) + 1                           # based on training steps; +1: final epoch 

    # learning rate scheduler
    n_warm_steps = int(total_steps/10)
    scheduler = WarmupLR(optimizer=optim, 
                        max_lr=[lr], 
                        num_warm=n_warm_steps, 
                        num_allsteps=total_steps,
                        decay_type=lr_decay_type)

    # mix
    if amp_on==True:
        scaler = GradScaler()  

    step_loss_collect = []
    step_acc_collect = []



    
    step_count = 0


    for epoch in range(n_epoch):
        model.train()
        for batch_data in tqdm(train_loader, desc='Pre-training',total=len(train_loader)):

            optim.zero_grad()
            
            sequences = batch_data['sequence'].values.ravel().tolist()
            #TODO: cell line label and classifier
           

            # slice sequences
            sliced_sequences = slice_sequences(sequences)
            
            if tokenization_mode == 'BPE_DNABERT':
                tokens_batch = [' '.join(Tokenizer.tokenize(sequence)) for sequence in sliced_sequences]
                maksed_tokens_batch = mask_dna_sequence(tokens_batch, mask_ratio=mask_ratio)
                maksed_tokens_batch = [masked_tokens.split() for masked_tokens in maksed_tokens_batch]
                inputs = Tokenizer.batch_encode_plus(maksed_tokens_batch, is_split_into_words=True, return_tensors='pt', padding=True).to(device)
                label_batch = Tokenizer(sliced_sequences, return_tensors="pt", padding=True).to(device) 
            else:
                # tokens: Base-level, BPE , k-mer
                tokens_batch = piece_sequences(sliced_sequences, tokenization_mode, sp=sp)   
                        
                # masking
                if (tokenization_mode == 'Base-level') or (tokenization_mode == 'BPE'):
                    maksed_tokens_batch = mask_dna_sequence(tokens_batch, mask_ratio=mask_ratio)
                else:
                    maksed_tokens_batch = mask_dna_sequence(tokens_batch, mask_ratio=mask_ratio, continuous_tokens=continuous_mask_tokens)

                # tokens -> inputs
                inputs = Tokenizer(maksed_tokens_batch, return_tensors="pt", padding=True).to(device)
                label_batch = Tokenizer(tokens_batch, return_tensors="pt", padding=True).to(device)  
                
 
            # model forward
            with autocast(device_type=device):
                outputs = model(**inputs)     
                
            logit = outputs.logits

            # loss
            mask_positions = (inputs.input_ids == Tokenizer.mask_token_id)
            mask_logits = logit.view(-1,  Tokenizer.vocab_size )[mask_positions.view(-1)]
            mask_labels = label_batch.input_ids.view(-1)[mask_positions.view(-1)]
            loss = criterion(mask_logits.to(device), mask_labels.to(device))
            loss = torch.mean(loss) 
            step_loss_collect.append(loss.cpu().item())
            
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            
            

            # accuracy 
            acc = compute_metrics(logit, label_batch.input_ids, mask_positions)
            step_acc_collect.append(acc)
            
            # learning rate warmup & decay
            scheduler.step()
                
            # monitoring    
            if args.wandb:
                wandb.log({'learning_rate': optim.param_groups[0]['lr'], 'step_loss': loss, 'step_acc': acc})
            
            
            # step counting 
            step_count+=1
            if isinstance(check_point, int):
                if (step_count % check_point == 0) | (step_count >= total_steps):  
                    model.save_pretrained(model_dir) # saving model weights
                    

            # final pre-training step
            if step_count  >= total_steps: 
                model.save_pretrained(model_dir) # saving model weights
                
                save_txt_single_column(step_loss_collect, filename='train_step_loss.txt', save_dir=save_dir)
                save_txt_single_column(step_acc_collect, filename='train_step_acc.txt', save_dir=save_dir)
                return True
             


    



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # pre-training params setting 
    parser.add_argument('--batch_size', required=True, type=int, help="Pre-training mini-batch size, default:256.")
    parser.add_argument('--total_steps', required=False, default=10000, type=int, help="The number of total pre-training steps.")
    parser.add_argument('--lr', required=True, type=float, help="Maximum learning rate.")
    parser.add_argument('--lr_decay_type', required=False,  default='cosine',type=str, help=".")
    parser.add_argument('--max_len_tokens', required=False, default=512, type=int, help="Maximum number of tokens in a sequence.")
    parser.add_argument('--mask_ratio', required=True, default=0.15, type=float, help="The ratio of masked k-mers in a sequence; defualt 0.15.")
    parser.add_argument('--continuous_mask_tokens', required=False, default=5, type=int, help=".")
    parser.add_argument('--check_point', required=False, default=2000, type=int, help=".")
    
    
    # model params setting 
    parser.add_argument('--tokenization', required=False, default="4_mer", type=str, help=".")
    parser.add_argument('--n_heads', required=False, default=12, type=int, help=".")
    parser.add_argument('--n_layers', required=False, default=12, type=int, help=".")
    parser.add_argument('--d_model', required=False, default=3072, type=int, help=".")
    
    # path
    parser.add_argument('--save_dir', required=True,type=str, help="The saving path of the trained model")
    parser.add_argument('--pretrain_data_path', required=True, type=str, help="The path of the pret-raining cell line atac-seq data")

    # tool setting
    parser.add_argument('--wandb', default=False, type=bool, help="Turn on the tool for visualization of training progress")
    parser.add_argument('--use_gpu', default=False, type=bool, help="Turn on to use gpu to train model; False: using cpu, True: using gpu if available.")
    args = parser.parse_args()



    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    if args.wandb:
        trace_tool = wandb.init(project="TRAFICA_Celllines")


    # data loading 
    train_data_path = os.path.join(args.pretrain_data_path,'Cellline_atac_train_2m8.h5')
    train_loader = load_dataset_from_h5(train_data_path, batch_size=args.batch_size)

    
    # vocab & tokenizer (avoid repeate)
    if args.tokenization == 'BPE_DNABERT':
        Tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        sp=None
    else:
        vocab_path = './Tokenizers/Vocab_Files'
        create_directory(vocab_path)
        
        vocab_save_path = os.path.join(vocab_path, f"{args.tokenization}.txt")
        if not os.path.exists(vocab_save_path):
            if args.tokenization == 'BPE':
                DNA_VocabularyGeneration(args.tokenization, save_path = vocab_save_path, 
                                        data_path=train_data_path,BPE_folder=os.path.join('./Tokenizers','BPE_Processing'))    # take a while; better to pre-process this step
            else:
                DNA_VocabularyGeneration(args.tokenization, save_path = vocab_save_path)
        else:
            print("Using the pre-constructed vocabulary.")
        

        if args.tokenization == 'BPE':
            sp = spm.SentencePieceProcessor()
            BPE_path = os.path.join('./Tokenizers','BPE_Processing', 'dna_BPE_model.model')
            if os.path.exists(BPE_path):
                sp.load(BPE_path)   
            else:
                raise FileNotFoundError(f"{BPE_path} not exists") 
        else:
            sp = None

        
        Tokenizer_save_path = os.path.join('./Tokenizers', args.tokenization)
        if not os.path.exists(Tokenizer_save_path):
            Tokenizer = BertTokenizerFast(vocab_save_path, do_lower_case=False, model_max_length=int(args.max_len_tokens+2)) # + 1 (CLS) +1 (SEP)
            Tokenizer.save_pretrained(Tokenizer_save_path)
        else:
            Tokenizer = BertTokenizerFast.from_pretrained(Tokenizer_save_path)

 
    # model
    configuration = RoFormerConfig(vocab_size=Tokenizer.vocab_size,
                                   hidden_size= args.d_model,
                                   num_hidden_layers=args.n_layers,
                                   num_attention_heads=args.n_heads,
                                   intermediate_size = int( 4*args.d_model ),
                                   rotary_value=True,
                                   max_position_embeddings=int(args.max_len_tokens+2),
                                   output_attentions=True,
                                   pad_token_id=Tokenizer.pad_token_id )
    model = RoFormerForMaskedLM(configuration)
    
   
    if torch.cuda.is_available():
        model.to(device)


    training_info = {'batch_size': args.batch_size,
                    'total_steps': args.total_steps,
                    'lr': args.lr,
                    'lr_decay_type': args.lr_decay_type,
                    'max_len_tokens': args.max_len_tokens,
                    'mask_ratio': args.mask_ratio,
                    'save_dir': args.save_dir,
                    'data_path': args.pretrain_data_path,
                    'continuous_mask_tokens':args.continuous_mask_tokens,
                    'wandb': args.wandb,
                    'use_gpu': args.use_gpu,
                    'num_model_params': sum([x.nelement() for x in model.parameters()]),
                    'data_size': len(train_loader.dataset),
                    'tokenization':args.tokenization,    
                    }
    
      
    create_directory(args.save_dir)
    save_json_file(os.path.join(args.save_dir, 'pretrain_params.json'),training_info)
    print(training_info)


    
    train(model=model,
        train_loader=train_loader,
        total_steps=args.total_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        mask_ratio=args.mask_ratio,
        tokenization_mode=args.tokenization,
        amp_on=True,
        lr_decay_type=args.lr_decay_type,
        continuous_mask_tokens=args.continuous_mask_tokens,
        check_point =args.check_point,
        sp=sp
        )
    
    



# CUDA_VISIBLE_DEVICES=0 python pretraining.py \
# --batch_size 128 \
# --total_steps 500000 \
# --lr 0.0001 \
# --max_len_tokens 512 \
# --mask_ratio 0.15 \
# --tokenization 4_mer \
# --continuous_mask_tokens 10 \
# --n_heads 8 \
# --n_layers 8 \
# --d_model 512 \
# --use_gpu true \
# --check_point 2000 \
# --save_dir ../Pretrained_TRAFICA/4_mer \
# --pretrain_data_path ../Data/Cellline_atac_seq
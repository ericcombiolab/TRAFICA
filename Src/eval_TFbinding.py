
from transformers import RoFormerConfig, AutoTokenizer, RoFormerForSequenceClassification 
import torch
from peft import PeftModel
import sklearn.metrics as metrics
from scipy.stats import pearsonr
import argparse
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





if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    # fine-tuning setting 
    parser.add_argument('--batch_size', required=True, type=int, help="Batch size.")
    parser.add_argument('--use_gpu', default=False, type=bool, help="Turn on to use gpu to train model; False: using cpu, True: using gpu if available.")

    # basic setting
    parser.add_argument('--predict_type', default='regression', type=str, help="TRAFICA support two types of prediction: regression and classification")

    # path
    parser.add_argument('--save_dir', required=True,type=str, help="The saving path of the finetuned model and/or evaluation result")
    parser.add_argument('--eval_data_path', required=True, type=str, help="The path of the training set file/folder")
    parser.add_argument('--tokenizer_path', required=True, type=str, help="The path of the tokenizer")
    parser.add_argument('--pretrained_model_path', required=False, default=None, type=str, help="The path of the pretrained model")
    parser.add_argument('--finetuned_lora_path', required=False, default=None, type=str, help="The path of the pretrained model")
    
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
    # test_loader= load_dataset_from_txt(os.path.join(args.eval_data_path, 'test.txt'), 
    #                                     shuffle=False,
    #                                     batch_size=args.batch_size)
    test_loader= load_dataset_from_txt(args.eval_data_path, 
                                        shuffle=False,
                                        batch_size=args.batch_size)

    
    # tokenizer
    if args.tokenization == 'BPE':
        sp = spm.SentencePieceProcessor()
        BPE_path = os.path.join('./Tokenizers','BPE_Processing', 'dna_BPE_model.model')
        sp.load(BPE_path)
    else:
        sp=None
    # else:
    #     Tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)    
    Tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)    
        
        
    # model    
    config = RoFormerConfig.from_pretrained(args.pretrained_model_path)
    config.num_labels = 1 
    test_model = RoFormerForSequenceClassification.from_pretrained(args.pretrained_model_path, config=config)
    state_dict = torch.load(os.path.join(args.finetuned_lora_path,"predict_head_weights.pth"),weights_only=True)  # 
    test_model.classifier.load_state_dict( state_dict['PREDICT_HEAD'] )
    test_model = PeftModel.from_pretrained(test_model, os.path.join(args.finetuned_lora_path,"lora_adapter"))
    test_model.to(device)
    
    
    # evaluation
    test_y, test_y_hat = testing(test_model, test_dataloader=test_loader, Tokenizer=Tokenizer, tokenization=args.tokenization, sp=sp, device=device)
    
    save_txt_single_column(test_y, args.save_dir, 'test_y.txt')
    save_txt_single_column(test_y_hat, args.save_dir, 'test_y_hat.txt')
    
    if args.predict_type == 'regression':
        evaluation_metric = ['r2', 'pcc', 'mse']
    elif args.predict_type == 'classification':
        evaluation_metric = ['auroc', 'auprc']
    else:
        raise TypeError('Evaluation error: TRAFICA support two types of prediction: regression and classification')

    test_y_min = min(test_y)
    test_y_max = max(test_y)
    test_y_norm = [(y - test_y_min) / (test_y_max - test_y_min) for y in test_y]
    
    for metric in evaluation_metric:
        metric_score = compute_metrics(test_y_norm, test_y_hat, evaluation_metric=metric)    
        save_txt_single_column([metric_score], args.save_dir, f"{metric}.txt")
    
    


## evaluating HT-SELEX fine-tuned models on PBM datasets
    
# CUDA_VISIBLE_DEVICES=1 python eval_TFbinding.py \
# --batch_size 128 \
# --use_gpu true \
# --save_dir ../Eval_TRAFICA/HTSELEX_PBM_Evaluation/FOXC2_TGAGTG30NTGA_Foxc2_pTH3796_HK  \
# --eval_data_path ../Data/2_Eval_HTSELEX_PBM/Foxc2_pTH3796_HK.txt \
# --tokenizer_path ./Tokenizers/Base-level \
# --tokenization Base-level \
# --predict_type regression \
# --pretrained_model_path ../Pretrained_TRAFICA/medium_Base-level/TRAFICA_Weights \
# --finetuned_lora_path ../Finetuned_TRAFICA_All/Base-level/PRJEB14744/10000/FOXC2_TGAGTG30NTGA     


## evaluating HT-SELEX fine-tuned models on ChIP-seq datasets

# CUDA_VISIBLE_DEVICES=1 python eval_TFbinding.py \
# --batch_size 128 \
# --use_gpu true \
# --save_dir ../Eval_TRAFICA/HTSELEX_ChIP_Evaluation/E2F2_TACTCA20NCG_E2F2_HepG2 \
# --eval_data_path ../Data/3_Eval_HTSELEX_ChIP/E2F2_HepG2.txt \
# --tokenizer_path ./Tokenizers/Base-level \
# --tokenization Base-level \
# --predict_type classification \
# --pretrained_model_path ../Pretrained_TRAFICA/medium_Base-level/TRAFICA_Weights \
# --finetuned_lora_path ../Finetuned_TRAFICA_All/Base-level/PRJEB14744/10000/E2F2_TACTCA20NCG     



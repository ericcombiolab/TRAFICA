import numpy as np
from transformers import RoFormerConfig, AutoTokenizer, RoFormerForSequenceClassification 
import torch
from peft import PeftModel
import numpy as np
import argparse
from TRAFICA.dataset import load_dataset_from_txt
from tqdm import tqdm 
from util import *


def compute_avg_attn_vectors(attn_heads: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute average attention vectors for each head across sequences.
    
    Args:
        attn_heads: List of attention matrices for n heads, each of shape 
                    (num_seqs, seq_len, seq_len)
                    
    Returns:
        List of average attention vectors for each head, each of shape (seq_len,)
    """
    avg_vectors = []
    for head_matrices in attn_heads:
        # Average across sequences and source tokens to get target token attention
        avg_vector = np.mean(head_matrices, axis=(0, 1))
        avg_vectors.append(avg_vector)
    return avg_vectors


def calculate_gini(vectors):
    """
    parameters::
        vectors: [batch, num_vectors, seq_len] or [num_vectors, seq_len]
    return:
        gini:  [batch, num_vectors] or [num_vectors]
    """
    if vectors.dim() == 2:
        vectors = vectors.unsqueeze(0)
        
    batch_size, num_vectors, seq_len = vectors.shape
    device = vectors.device
    

    vectors = vectors + 1e-8
    
    
    sorted_vec, _ = torch.sort(vectors, dim=-1)  
    cumsum = torch.cumsum(sorted_vec, dim=-1)  
    
    
    idx = torch.arange(1, seq_len + 1, device=device).float()
    numerator = torch.sum(cumsum, dim=-1)  
    denominator = torch.sum(vectors, dim=-1)  
    gini = (seq_len + 1) / seq_len - 2 * numerator / (seq_len * denominator + 1e-8)
    
    return gini.squeeze(0) if batch_size == 1 else gini


def select_important_heads(attn, n_topHeads=2):
    """
    parameters:
        last_layer_attn: [batch, num_heads, seq_len, seq_len]
        n_topHeads: num of heads
    return:
        top_head_indices: idx of top heads [n_topHeads]
        head_gini: averge gini of all heads [num_heads]
    """
    device = attn.device
    num_heads = attn.shape[1]
    
    # 
    avg_vectors = torch.mean(attn, dim=0)
    
    # [batch, heads]
    gini = calculate_gini(avg_vectors)

    
    # obtain n_topHeads
    top_head_indices = torch.argsort(gini, descending=True)[:n_topHeads]
    
    return top_head_indices, gini



def attn_extracting(model, test_dataloader, Tokenizer, tokenization, sp, t_layers, n_seqs, device):
    model.eval() # switch the model to evaluation mode
    attns_collect = []  
    pred_collect = []  
    seq_collect = []

    for data in tqdm(test_dataloader, desc='attn extracting...',total=len(test_dataloader)):
        
        sequences, y = data
        if tokenization == 'BPE_DNABERT':
            inputs = Tokenizer(sequences, return_tensors='pt', padding=True).to(device)
        else:
            tokens_batch = piece_sequences(sequences, tokenization, sp)   
            inputs = Tokenizer(tokens_batch, return_tensors="pt", padding=True).to(device)
        
        inputs['output_attentions'] = True                          # return attn
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logit = outputs.logits
        pred_collect += logit.flatten().cpu().detach().numpy().tolist()

        attns = outputs.attentions[-t_layers:]                      # last N layers
        attns = torch.stack(attns, dim=1)                           # stack in layer dim

        attns_received = attns.mean(dim=-2)                         # matrix -> vector
        attns_collect.append(attns_received) 

        seq_collect+=sequences

    top_seq_indices = np.argsort(np.array(pred_collect))[-n_seqs:][::-1]  
    
    attns_batch = torch.concat(attns_collect, dim=0).cpu().detach().numpy() # batch attention processing
    attns_selected = attns_batch[top_seq_indices]
    attns_selected= attns_selected[:,:,:,1:-1]                              # drop <cls> <sep>


    seqs_selected = np.array(seq_collect)[top_seq_indices]

    return attns_selected, seqs_selected, np.array(pred_collect)[top_seq_indices]
    




    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # fine-tuning setting 
    parser.add_argument('--batch_size', required=True, type=int, help="Batch size.")
    parser.add_argument('--use_gpu', default=False, type=bool, help="Turn on to use gpu to train model; False: using cpu, True: using gpu if available.")


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
    
    attns, seqs, pred = attn_extracting(test_model, 
                           test_dataloader=test_loader, 
                           Tokenizer=Tokenizer, 
                           tokenization=args.tokenization,
                           sp=sp, 
                           t_layers=6,
                           n_seqs=1000,
                           device=device)
    
    
    attns = np.mean(attns, axis=1)      # which transformer layers
    head_idx, gini = select_important_heads(torch.Tensor(attns), n_topHeads=8)
    atts_selectedHeads = attns[:,head_idx,:]
  
    np.savez_compressed(
        os.path.join(args.save_dir, f"attns.npz"),
        attns=atts_selectedHeads,    
        head_indices=head_idx,   
        gini_scores=gini,
        sequences=seqs,         
        predicted_affinity=pred,       
    )
 


# CUDA_VISIBLE_DEVICES=0 python attn_extraction.py \
# --batch_size 128 \
# --use_gpu true \
# --save_dir ../Motif_Attn/Base-level/PRJEB14744/10000/E2F2_TACTCA20NCG/Attentions \
# --eval_data_path ../Data/HT_SELEX/PRJEB14744/10000/E2F2_TACTCA20NCG/test.txt \
# --tokenizer_path ./Tokenizers/Base-level \
# --tokenization Base-level \
# --pretrained_model_path ../Pretrained_TRAFICA/medium_Base-level/TRAFICA_Weights \
# --finetuned_lora_path ../Finetuned_TRAFICA_All/Base-level/PRJEB14744/10000/E2F2_TACTCA20NCG     

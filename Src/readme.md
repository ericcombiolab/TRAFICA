**************
## 1. TRAFICA Pre-training
**Run the below command**
``` bash
CUDA_VISIBLE_DEVICES=0 python pretraining.py \
--batch_size 128 \
--total_steps 500000 \
--lr 0.0001 \
--max_len_tokens 512 \
--mask_ratio 0.15 \
--tokenization 4_mer \
--continuous_mask_tokens 10 \
--n_heads 8 \
--n_layers 8 \
--d_model 512 \
--use_gpu true \
--check_point 2000 \
--save_dir ../Pretrained_TRAFICA/4_mer \
--pretrain_data_path ../Data/Cellline_atac_seq
```

**Output**  
|_ Example_Pretrain_result  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ TRAFICA_Weights  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_  config.json  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_  generation_config.json  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_  model.safetensors 
|&nbsp;&nbsp;&nbsp;&nbsp;|_ pretrain_params.json  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ train_step_acc.txt  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ train_step_loss.txt  

**Tokenization Version**  
`--tokenization` can be `4_mer`, `5_mer`, `6_mer`, `Base-level`, `BPE`, and `BPE_DNABERT`  

+ `BPE`: vocabulary constructed using our ATAC-seq dataset
+ `BPE_DNABERT`:  vocabulary established in the DNABERT2 study


**Pre-training Data**  
`--pretrain_data_path`: path to the file `Cellline_atac_train_2m8.h5`, which is available at [Here]()




**************
## 2. Reproduction of Experimental Results
### 2.1 TRAFICA Fine-tuning on PBM/HT-SELEX Datasets
**Run the below command**
``` bash
python finetuning_TFbinding_AutoRun.py \
--mode PBM \
--tokenization Base-level \
```
 
+ Make sure these two data folders exist: `../Data/DREAM5_PBM_protocol` and `../Data/HT_SELEX`
+ --mode: `PBM`, `HT-SELEX`, or `Ablation_PreTrain`
+ --tokenization: `4_mer`, `5_mer`, `6_mer`, `Base-level`, `BPE`, or `BPE_DNABERT`  

### 2.2 TRAFICA Assessment on HT-SELEX Evaluation Protocol
**Run the below command**
``` bash
python finetuning_TFbinding_AutoRun.py \
--mode Cross-platform \
--tokenization Base-level \
```
+ Make sure these three files exist: `../Data/HT_SELEX_ChIP_overlap.csv`,`../Data/HT_SELEX_CrossExp_overlap.csv` and `../Data/HT_SELEX_PBM_DREAM5_overlap.csv`
+ --mode: `Cross-platform`, `Cross-experiment`, or `In-vivo`
+ --tokenization: `4_mer`, `5_mer`, `6_mer`, `Base-level`, `BPE`, or `BPE_DNABERT`  


### 2.3 Motif analysis
**Run the below command**
``` bash
python attn_extraction_AutoRun.py
python attn_motif_AutoRun.py
```
+ Make sure these folders exist: `../Data/HT_SELEX` and `../Finetuned_TRAFICA_All/Base-level`



**************
## 3. Fine-tuning TRAFICA Custom datasets
### 3.1 Fine-tuning 
**Run the below command**
``` bash
CUDA_VISIBLE_DEVICES=1 python finetuning_TFbinding.py \
--batch_size 128 \
--lr 0.0001 \
--n_epoch 300 \
--n_toler 10 \
--use_gpu true \
--save_dir  \
--eval_data_path \
--tokenizer_path  \
--tokenization  \
--pretrained_model_path       
```

+ `eval_data_path`: the folder contains `train.txt`, `val.txt`, and `test.txt`  
+ `tokenizer_path`: can be `./Tokenizers/4mer` ...
+ `tokenization`: corresponding with `tokenizer_path`

**Output**  
|_ save_dir   
|&nbsp;&nbsp;&nbsp;&nbsp;|_ mse.txt  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ pcc.txt  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ r2.txt  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ test_y_hat.txt  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ test_y.txt  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ train_loss_epoch.txt  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ val_loss_epoch.txt  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ training_setting.txt  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ predict_head_weights.pth  
|&nbsp;&nbsp;&nbsp;&nbsp;|_ lora_adapter  
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|_ adapter_config.json  
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|_ adapter_model.safetensors  




**************



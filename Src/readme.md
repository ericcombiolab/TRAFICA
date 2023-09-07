# Model Implementation
The pre-trained model and hundreds of the trained adapters are released in the HuggingFace hub [Link](https://huggingface.co/Allanxu/TRAFICA/tree/main).

The overview of 
1. Model pre-training
2. Model fine-tuning  
    1. Fully fine-tuning
    2. Adapter-tuning
    3. AdapterFusion
3. Motif discovery (in sorting matrials...)
4. Speed up training with batch datasets using multiple GPUs (on one machine)
5. Notice


**************
## Pre-train
Here is an example of pre-training model in a single GPU
``` bash
python pretraining.py \
--batch_size 384 \
--total_training_step 110000 \
--lr_warmup 10000 \
--lr 0.0001 \
--mask_ratio 0.15 \
--mask_n_phrases 10 \
--use_gpu True \
--model_save_dir ../Example_Pretrain_result \
--train_data_path ../Example_Pretrain_DataFolder \
--vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt
```
TODO: revise HuggingFace model input to support multi-gpu pre-training.


**************


## Fully fine-tuning
### Step 1. Loading pre-trained model and fine-tuning
Run the following commend to fine-tune the pre-trained model on the example dataset
``` bash
CUDA_VISIBLE_DEVICES=3 python finetuning.py \
--batch_size 128 \
--n_epoches 50 \
--n_epoches_tolerance 5 \
--lr 0.00002 --k 4 \
--save_dir ../Examples/Example_FT_result/RFX5_TGGAGC30NGAT \
--data_path ../Examples/Example_data/RFX5_TGGAGC30NGAT_4mer_finetuneDataFolder/RFX5_TGGAGC30NGAT_4mer \
--vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
--pretrained_model_path Allanxu/TRAFICA \
--use_gpu True \
--task_type regression \
--pool_strategy mean \
--name_experiment RFX5_TGGAGC30NGAT \
--save_dir_metric ../Examples/Example_FT_result 
```
> The output of the above example
>> ../Examples/Example_FT_result/RFX5_TGGAGC30NGAT

### Step 2. Making prediction by using the fully fine-tuned model (In vitro)
``` bash
CUDA_VISIBLE_DEVICES=3 python make_prediction.py \
--data_path ../Examples/Example_data/RFX5_TGGAGC30NGAT_4mer_finetuneDataFolder/RFX5_TGGAGC30NGAT_4mer/test.4mer \
--model_dir ../Examples/Example_FT_result/RFX5_TGGAGC30NGAT \
--save_dir ../Examples/Predict_FT_result \
--vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
--use_gpu True \
--pool_strategy mean \
--task_type regression \
--evaluation_metric set2 \
--batch_size 32
```
> The output of the above example
>> ../Examples/Predict_FT_result/
>>> `pcc.txt`,  `r2.txt`, `test_y_hat.txt`, `test_y.txt`



## Adapter-tuning
### Step 1. Loading pre-trained model and fine-tuning with adapter modules
In this phase, the parameters of the pre-trained model are frozen. **Users can directly adopt our trained adapters and skip adapter-tuning procedures.**

``` bash
CUDA_VISIBLE_DEVICES=3 python finetuning.py \
--batch_size 128 \
--n_epoches 50 \
--n_epoches_tolerance 5 \
--lr 0.00002 --k 4 \
--save_dir ../Examples/Example_AD_result/RFX5_TGGAGC30NGAT \
--data_path ../Examples/Example_data/RFX5_TGGAGC30NGAT_4mer_finetuneDataFolder/RFX5_TGGAGC30NGAT_4mer \
--vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
--pretrained_model_path Allanxu/TRAFICA \
--use_gpu True \
--task_type regression_adapter \
--pool_strategy mean \
--name_experiment RFX5_TGGAGC30NGAT \
--save_dir_metric ../Example_AD_result
```
> The output of the above example
>> ../Examples/Example_AD_result/RFX5_TGGAGC30NGAT

* We already trained hundreds of adapters using HT-SELEX and ChIP-seq datasets, which have been released in the HuggingFace model hub. These adapters can be used together with our pre-trained model to perform AdapterFusion precedures or make prediction directly. The list of the trained adapters is in (Link)

* Training ChIP-seq adapters is the same process, modifying the params:   
(1) `--task_type regression_adapter` -> `--task_type classification_adapter`;   
(2) `--lr 0.00002` -> `--lr 0.0001`    
(3) Data path and Save path corresponding to the specific dataset


### Step 2. Making prediction by using the pre-trained model coupled with the trained adapeters (In vitro)
``` bash
CUDA_VISIBLE_DEVICES=3 python make_prediction.py \
--data_path ../Examples/Example_data/RFX5_TGGAGC30NGAT_4mer_finetuneDataFolder/RFX5_TGGAGC30NGAT_4mer/test.4mer \
--model_dir ../Examples/Example_AD_result/RFX5_TGGAGC30NGAT \ 
--save_dir ../Examples/Predict_AD_result \
--vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
--use_gpu True \
--pretrained_model_path  Allanxu/TRAFICA \
--pool_strategy mean \
--task_type regression_adapter \
--evaluation_metric set2 \
--batch_size 32
```
> The output of the above example
>> ../Examples/Predict_AD_result/
>>> `pcc.txt`,  `r2.txt`, `test_y_hat.txt`, `test_y.txt`




## AdapterFusion
### Step 1. Generating an adapter map for a specific HT-SELEX adapter  
* In the example of [the above adapter-tuning](#adapter-tuning), we train a HT-SELEX adapter on RFX5_TGGAGC30NGAT dataset. Here we construct a map that incorperates [ChIP-seq adapters](https://huggingface.co/Allanxu/TRAFICA/tree/main/ChIP_seq_finetuned_adapter) trained on other TFs' data (excluding TF RFX5) to enhance in vivo predictive performance.
* The map is stored in a JSON format, in which each key-value pair represents a trained adapter. The keys are the names of datasets used to train adapters, and the values are the path of the trained adapters. For RFX5_TGGAGC30NGAT, one example of the adapter map in my machine environment is provided in [`PRJEB3289_RFX5_TGGAGC30NGAT.json`](./adapter_maps/PRJEB3289_RFX5_TGGAGC30NGAT.json). The values in the map should be matched with user's local path of the HT-SELEX and ChIP-seq adapters.


### Step 2. Loading pre-trained model and trained adapter modules, tuning the AdapterFusion module
In this phase, the parameters of the pre-trained model and the adapter modules are frozen. 

``` bash
CUDA_VISIBLE_DEVICES=2 python finetuning.py \
--batch_size 4 \
--n_epoches 50 \
--n_epoches_tolerance 3 \
--lr 0.0001 --k 4 \
--save_dir ../Examples/Example_AF_result/RFX5_TGGAGC30NGAT \
--data_path ../Examples/Example_data/RFX5_TGGAGC30NGAT_4mer_finetuneDataFolder/RFX5_TGGAGC30NGAT_4mer \
--vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
--pretrained_model_path Allanxu/TRAFICA \
--use_gpu True \
--task_type regression_adapterfusion \
--pool_strategy mean \
--name_experiment Example_RFX5_TGGAGC30NGAT_adapterfusion \
--save_dir_metric ../Examples/Example_AF_result \
--pretrained_adapters_path ./adapter_maps/PRJEB3289_RFX5_TGGAGC30NGAT.json \
--max_num_valsample 1000 \
--max_num_trainsample 2000
```
> The output of the above example
>> ../Examples/Example_AF_result/RFX5_TGGAGC30NGAT


### Step 3. Making prediction by using the combination of the pre-trained model, the trained adapeters, and the AdapterFusion module (In vitro and in vivo)

* In vivo evaluation
``` bash
CUDA_VISIBLE_DEVICES=0 python make_prediction.py \
--data_path ../Examples/Example_data/RFX5_GM12878/RFX5_GM12878_RFX5_\(200-401-194\)_Stanford_4mer/test.4mer \
--model_dir ../Examples/Example_AF_result/RFX5_TGGAGC30NGAT \
--save_dir ../Examples/Predict_AF_result_invivo \
--vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
--use_gpu True \
--pretrained_model_path Allanxu/TRAFICA \
--pool_strategy mean \
--task_type regression_adapterfusion \
--evaluation_metric set1 \
--pretrained_adapters_path ./adapter_maps/PRJEB3289_RFX5_TGGAGC30NGAT.json \
--batch_size 2
```
> The output of the above example
>> ../Examples/Predict_AF_result_invivo/
>>> `auroc.txt`, `test_y_hat.txt`, `test_y.txt`

* In vitro evaluation
``` bash
CUDA_VISIBLE_DEVICES=0 python make_prediction.py \
--data_path ../Examples/Example_data/RFX5_TGGAGC30NGAT_4mer_finetuneDataFolder/RFX5_TGGAGC30NGAT_4mer/test.4mer \
--model_dir ../Examples/Example_AF_result/RFX5_TGGAGC30NGAT \
--save_dir ../Examples/Predict_AF_result_invitro \
--vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt \
--use_gpu True \
--pretrained_model_path Allanxu/TRAFICA \
--pool_strategy mean \
--task_type regression_adapterfusion \
--evaluation_metric set2 \
--pretrained_adapters_path ./adapter_maps/PRJEB3289_RFX5_TGGAGC30NGAT.json \
--batch_size 6
```
>> ../Examples/Predict_AF_result_invitro/
>>> `pcc.txt`,  `r2.txt`, `test_y_hat.txt`, `test_y.txt`



**************

## Motif discovery based on attention scores
sorting codes and materials ~~~~


## Training on batch datasets on multiple GPUs to speed up the progresses
1. `transfer_sever.py`: environment requirement: expect & tcl (two common linux tools); [expect_scp.exp](../expect_scp.exp)
2. `parallel_recoder.py`


## Accessing the released pre-trained model and the trained adapters
We recommended a most easy way to access the trained parameters, but requiring the Internet connection. Just set the param `--pretrained_model_path ` as the following: 
``` bash
--pretrained_model_path Allanxu/TRAFICA 
```

If these is not Internet connection for your local environment, you can download the pre-trained model from [HuggingFace_released_v1](https://huggingface.co/Allanxu/TRAFICA/blob/main/pytorch_model.bin) by using other machine with network accessiblity, and change the param of `--pretrained_model_path` replaced by the local path of the model.


## Notice
We just provided the pre-trained model and the trained adapters for users, because the fully fine-tuned models and the AdapterFusion modules require a lot of storage space to save them.   

User can easy load the pre-trained model to fine-tune with their own data. In addition, the trained adapters are provided for user to perform AdapterFusion for in vivo binding affinities prediction.

Moreover, the pre-trained model coupled with the trained adapters also can be used to predict in vitro relative binding affinities directly. We have evaluated its performance and obtained the [promising results](https://huggingface.co/Allanxu/TRAFICA/blob/main/PRJEB3289_finetuned_adapter/test_pcc.json).  



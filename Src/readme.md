# Model Implementation
The pre-trained model and hundreds of the fine-tuned adapters are released in the HuggingFace hub [Link](https://huggingface.co/Allanxu/TRAFICA/tree/main).



**************
## Pre-train
Here is an example of pre-training model with example ATAC-seq data in a single GPU
``` bash
CUDA_VISIBLE_DEVICES=1 python pretraining.py \
--batch_size 384 \
--total_training_step 110000 \
--lr_warmup 10000 \
--lr 0.0001 \
--mask_ratio 0.15 \
--mask_n_phrases 10 \
--use_gpu True \
--model_save_dir ../Example_Pretrain_result \
--train_data_path ../ExampleData/ATAC_seq \
--vocab_path ../vocab_k_mer/vocab_DNA_4_mer.txt
```
TODO: revise HuggingFace model input to support multi-gpu pre-training.


**************


## Fine-tuning
###  Fully fine-tuning: Loading pre-trained model and fine-tuning
Run the following commend to fine-tune the pre-trained model on the example dataset
``` bash
CUDA_VISIBLE_DEVICES=1 python finetuning.py \
--batch_size 128 \
--n_epoches 50 \
--n_epoches_tolerance 5 \
--lr 0.00002 \
--save_dir ../ExampleFullyFineTune/HT_SELEX/RFX5_TGGAGC30NGAT \
--train ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/train.txt \
--val ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/val.txt \
--test ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/test.txt \
--pretrain_tokenizer_path ./tokenizer \
--pretrained_model_path Allanxu/TRAFICA \
--use_gpu True \
--task_type FullyFineTuning \
--predict_type regression \
--inputfile_types DataMatrix \
--save_dir_metric ../ExampleFullyFineTune/HT_SELEX \
--name_experiment RFX5_TGGAGC30NGAT 
```



### Adapter-tuning: Loading pre-trained model and fine-tuning with adapters
In this phase, the parameters of the pre-trained model are frozen. **Users can directly adopt our fine-tuned adapters and skip adapter-tuning procedures.**

``` bash
CUDA_VISIBLE_DEVICES=1 python finetuning.py \
--batch_size 128 \
--n_epoches 50 \
--n_epoches_tolerance 5 \
--lr 0.00002 \
--save_dir \
--train ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/train.txt \
--val ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/val.txt \
--test ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/test.txt \
--pretrain_tokenizer_path ./tokenizer \
--pretrained_model_path Allanxu/TRAFICA \
--use_gpu True \
--task_type AdapterTuning \
--predict_type regression \
--inputfile_types DataMatrix \
--save_dir_metric ../Examples/Example_AD_result_newversion \
--name_experiment RFX5_TGGAGC30NGAT 
```
specify the `save_dir`


* We already trained hundreds of adapters using HT-SELEX and ChIP-seq datasets, which have been released in the HuggingFace model hub. These adapters can be used together with our pre-trained model to perform AdapterFusion precedures or make prediction directly. The list of the fine-trained adapters is in [TRAFICA](https://huggingface.co/Allanxu/TRAFICA/tree/main)

* Training ChIP-seq adapters is the same process, modifying the params:   
(1) `--predict_type regression` -> `--predict_type classification`;   
(2) `--lr 0.00002` -> `--lr 0.0001`    
(3) Data path and Save path corresponding to the specific dataset


### AdapterFusion
#### Step (1) Generating an adapter map for a specific HT-SELEX adapter  
* In the example of the above adapter-tuning phase, we fine-tune a HT-SELEX adapter on RFX5_TGGAGC30NGAT dataset. Here we construct a map that incorperates TF-DNA adapters fine-tuned on other datasets (excluding TF RFX5) to enhance in vivo predictive performance. (e.g., [link](../AdapterMapsForFusion/ChIP_seq_137/PRJEB3289_RFX5_TGGAGC30NGAT.json))
* The map is stored in a JSON format, in which each key-value pair represents a fine-tuned adapter. The keys are the names of datasets used to train adapters, and the values are the path of the fine-tuned adapters. 
* Make sure the folders of "Adapters" downloaded from [HuggingFace](https://huggingface.co/Allanxu/TRAFICA/tree/main) and place in the correct path.

#### Step (2) Loading the pre-trained model and fine-tuned adapters, tuning the AdapterFusion module
In this phase, the parameters of the pre-trained model and the fine-tuned adapters are frozen. 

``` bash
CUDA_VISIBLE_DEVICES=1 python finetuning.py \
--batch_size 8 \
--n_epoches 50 \
--n_epoches_tolerance 3 \
--lr 0.0001 \
--save_dir  \
--train ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/train.txt \
--val ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/val.txt \
--test ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/test.txt \
--pretrain_tokenizer_path ./tokenizer \
--pretrained_model_path Allanxu/TRAFICA \
--finetuned_adapterlist_path ../AdapterMapsForFusion/ChIP_seq_137/PRJEB3289_RFX5_TGGAGC30NGAT.json \
--use_gpu True \
--task_type AdapterFusion \
--predict_type regression \
--inputfile_types DataMatrix \
--save_dir_metric ../ExampleAdapterFusion/HT_SELEX \
--name_experiment RFX5_TGGAGC30NGAT \
--max_num_val 1000 \
--max_num_train 2000
```
specify the `save_dir`


## Prediction
### Making prediction by using the fully fine-tuned model (In vitro)
This example is based on the model fine-tuned in the above procedure of Fully fine-tuning
``` bash
CUDA_VISIBLE_DEVICES=1 python prediction.py \
--data_path ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/test.txt \
--inputfile_types DataMatrix \
--save_dir ../PredictionExample \
--pretrain_tokenizer_path ./tokenizer \
--use_gpu True \
--finetuned_fullmodel_path ../ExampleFullyFineTune/HT_SELEX/RFX5_TGGAGC30NGAT \
--task_type FullyFineTuning \
--evaluation_metric set2 \
--batch_size 32
```


### Making prediction by using the pre-trained model coupled with the fine-tuned adapeters (In vitro)
``` bash
CUDA_VISIBLE_DEVICES=1 python prediction.py \
--data_path ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/test.txt \
--inputfile_types DataMatrix \
--save_dir  \
--pretrain_tokenizer_path ./tokenizer \
--use_gpu True \
--pretrained_model_path Allanxu/TRAFICA \
--finetuned_adapter_path ../Adapters/TF_DNA_Adapters/PRJEB3289/RFX5_TGGAGC30NGAT \
--task_type AdapterTuning \
--evaluation_metric set2 \
--batch_size 32
```
specify the `save_dir`


### Making prediction by using the combination of the pre-trained model, the fine-tuned adapeters, and the AdapterFusion component (In vitro and in vivo)

* In vitro
``` bash
CUDA_VISIBLE_DEVICES=1 python make_prediction.py \
--data_path ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/test.txt \
--inputfile_types DataMatrix \
--save_dir  \
--pretrain_tokenizer_path ./tokenizer \
--use_gpu True \
--pretrained_model_path  Allanxu/TRAFICA \
--finetuned_adapterfusion_path ../AdapterFusion/PRJEB3289_RFX5_TGGAGC30NGAT/ChIP_seq_All_137 \
--finetuned_adapterlist_path ../AdapterMapsForFusion/ChIP_seq_137/PRJEB3289_RFX5_TGGAGC30NGAT.json \
--task_type AdapterFusion \
--evaluation_metric set2 \
--batch_size 2
```
specify the `save_dir`


* In vivo
``` bash
CUDA_VISIBLE_DEVICES=1 python make_prediction.py \
--data_path "../ExampleData/ChIP_seq/RFX5_GM12878_RFX5_(200-401-194)_Stanford/test.seqlabel" \
--inputfile_types DataMatrix \
--save_dir \
--pretrain_tokenizer_path ./tokenizer \
--use_gpu True \
--pretrained_model_path  Allanxu/TRAFICA \
--finetuned_adapterfusion_path ../AdapterFusion/PRJEB3289_RFX5_TGGAGC30NGAT/ChIP_seq_All_137 \
--finetuned_adapterlist_path ../AdapterMapsForFusion/ChIP_seq_137/PRJEB3289_RFX5_TGGAGC30NGAT.json \
--task_type AdapterFusion \
--evaluation_metric set1 \
--batch_size 2
```
specify the `save_dir`




**************
## Motif discovery based on attention scores
sorting codes and materials ~~~~



## Accessing the released pre-trained model and the trained adapters
We recommended a most easy way to access the trained parameters, but requiring the Internet connection. Just set the param `--pretrained_model_path ` as the following: 
``` bash
--pretrained_model_path Allanxu/TRAFICA 
```

If these is not Internet connection for your local environment, you can download the pre-trained model from [HuggingFace_released_v1](https://huggingface.co/Allanxu/TRAFICA/blob/main/pytorch_model.bin) by using other machine with network accessiblity, and change the param of `--pretrained_model_path` replaced by the local path of the model.


## Notice
We just provided the pre-trained model and the fine-tuned adapters for users, because the fully fine-tuned models and the AdapterFusion component requiring a little bit more disk storage to save them.   

User can easy load the pre-trained model to fine-tune with their own data. In addition, the fine-tuned adapters are provided for user to perform AdapterFusion for in vivo binding affinities prediction.

Moreover, the pre-trained model coupled with the fine-tuned adapters also can be used to predict in vitro relative binding affinities directly.  

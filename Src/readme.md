# TRAFICA pre-training, fine-tuning, and inference

**************
## TRAFICA pre-training
Here is an example of pre-training TRAFICA with example ATAC-seq data in a single GPU
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


**************


## TRAFICA fine-tuning
Run the following commend to fine-tune TRAFICA on the example dataset
``` bash
CUDA_VISIBLE_DEVICES=1 python finetuning.py \
--batch_size 128 \
--n_epoches 50 \
--n_epoches_tolerance 5 \
--lr 0.00002 \
--save_dir ../Example_FineTune_result/HT_SELEX/RFX5_TGGAGC30NGAT \
--train ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/train.txt \
--val ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/val.txt \
--test ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/test.txt \
--pretrain_tokenizer_path ./tokenizer \
--pretrained_model_path Allanxu/TRAFICA \
--use_gpu True \
--task_type FullyFineTuning \
--predict_type regression \
--inputfile_types DataMatrix \
--save_dir_metric ../Example_FineTune_result/HT_SELEX \
--name_experiment RFX5_TGGAGC30NGAT 
```

These two parameters should be added if users want to generate subsets for model fine-tuning
``` bash
--max_num_train 10000
--max_num_val 1000
```
The above example can generate a subset with 10000 sequences for fine-tuning. The generated susbset can be found in the path `../Example_FineTune_result/HT_SELEX/RFX5_TGGAGC30NGAT/subset_10000`.

**************
## TRAFICA inference
### Making prediction by using the fully fine-tuned model (In vitro)
This example is based on the model fine-tuned in the above procedure of Fully fine-tuning
``` bash
CUDA_VISIBLE_DEVICES=1 python prediction.py \
--data_path ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/test.txt \
--inputfile_types DataMatrix \
--save_dir ../Example_Inference \
--pretrain_tokenizer_path ./tokenizer \
--use_gpu True \
--finetuned_fullmodel_path ../Example_FineTune_result/HT_SELEX/RFX5_TGGAGC30NGAT \
--task_type FullyFineTuning \
--evaluation_metric set2 \
--batch_size 32
```



**************

**************

**************
## TRAFICA adapter-tuning (Additional feature)
> We also provide a parameter-efficient fine-tuning approarch to **save disk memory**. 
### Adapter-tuning
1. Insert the module of adapters in the Transformer-encoder architecture
2. Froze the parameters of the pre-trained model
3. Fine-tune the adapters

``` bash
CUDA_VISIBLE_DEVICES=1 python finetuning.py \
--batch_size 128 \
--n_epoches 50 \
--n_epoches_tolerance 5 \
--lr 0.00002 \
--save_dir ../Example_Adapter_result/HT_SELEX/RFX5_TGGAGC30NGAT \
--train ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/train.txt \
--val ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/val.txt \
--test ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/test.txt \
--pretrain_tokenizer_path ./tokenizer \
--pretrained_model_path Allanxu/TRAFICA \
--use_gpu True \
--task_type AdapterTuning \
--predict_type regression \
--inputfile_types DataMatrix \
--save_dir_metric ../Example_Adapter_result/HT_SELEX \
--name_experiment RFX5_TGGAGC30NGAT 
```

4. Coupling the pre-trained model with the fine-tuned adapters to make TF-DNA binding affinity prediction  


``` bash
CUDA_VISIBLE_DEVICES=1 python prediction.py \
--data_path ../ExampleData/HT_SELEX/RFX5_TGGAGC30NGAT/test.txt \
--inputfile_types DataMatrix \
--save_dir ../Example_Inference_Adapter \
--pretrain_tokenizer_path ./tokenizer \
--use_gpu True \
--pretrained_model_path Allanxu/TRAFICA \
--finetuned_adapter_path ../Example_Adapter_result/HT_SELEX/RFX5_TGGAGC30NGAT \
--task_type AdapterTuning \
--evaluation_metric set2 \
--batch_size 32
```

**************


# Quickly apply 
We released the weights of the pre-trained open chromatin language model and the fine-tunde adapters for hundreds of TFs (using HT-SELEX data). Users can directly apply TRAFICA to predict TF-DNA binding affinity without any training procedures (pre-train/fine-tune). 

## The weights of the pre-trained open chromatin model
Users can easily load and fine-tune the pre-trained TRAFICA on their private data by setting the parameter of `--pretrained_model_path` as follows: 
``` bash
--pretrained_model_path Allanxu/TRAFICA 
```

## The weights of the fine-tuned adapters
The list of the fine-trained adapters is available at [the HuggingFace repository](https://huggingface.co/Allanxu/TRAFICA/tree/main/Adapters)


**************



# TODO: add an explanation for each parameter of the input console command
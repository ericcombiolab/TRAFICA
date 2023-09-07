# TRAFICA:

## Code environment  


## Quick start (example) 
### Fully fine-tuning version
1. Git clone or download this repo to your local disk  
```
git clone git@github.com:yuxu-1/TRAFICA.git
```
2. [Data preprocessing](#sequences-to-k-mer-format-transformation): use the processed [example data](./Examples/Example_data/RFX5_TGGAGC30NGAT_4mer_finetuneDataFolder/) to fine-tune TRAFICA

3. Download the pre-trained model to local disk or load from online hub
> we released the pre-trained model in HuggingFace, which gave an easy way to access the model. (add link to model access section)

4. [Fine-tune the pre-trained model using the example data](./Src/readme.md#fully-fine-tuning)

5. Motif discovery (Sorting materials...)


### Adapter version
The step 1 to 3 is same with fully fine-tuning  

4. [Train the adapter and evaluate the model using the example data](./Src/readme.md#adapter-tuning)

5. [Train the AdapterFusion module and evaluate the model using the example data](./Src/readme.md#adapterfusion)


## Data availability 
* Raw data source

* Processed data


******************

## [Source code](./Src/)

 
## Tools for k-mer vocab construction, sequence2kmer, and oneline files

### Interface introduction:  
`python utils.py --help`
   


### K-mer vocabulary construction:  
**Example:** `python utils.py --vocab_construct True --k 4 --save_dir ./vocab_k_mer`
* --vocab_construct : set `True` to enable this operation;
* --k : the hyperparameter of `k`-mer
* --save_dir: the saving path


 
### Sequences to k-mer format transformation:  
**Example 1. (ATAC-seq data for pre-train):** `python utils.py --seq_2_kmer True --sequence_dir /tmp/csyuxu/processed_ATAC_seq_ENCODE_p100 --k 4`  
**Example 2. (HT-SELEX data for fine-tune (single experimental data)):** `python utils.py --seq_2_kmer True --sequence_dir ./Examples/Example_data/RFX5_TGGAGC30NGAT --k 4 --input_label True`  
**Example 3. (HT-SELEX data for fine-tune (Batch experimental data)):** `python utils.py --seq_2_kmer True --sequence_dir /tmp/csyuxu/PRJEB3289 --k 4 --input_label True  --level_sub_folder 2`
* --seq_2_kmer : set `True` to enable this operation;
* --k : the hyperparameter of `k`-mer
* --sequence_dir: the path of sequence file(s)
* --input_label: set `True` if each sequene has a label in last column
* --level_sub_folder: subfolder level (e.g. `2`: /XXsaveXX/XXexpXX/ , each independent experiment has a folder; `1` (default): /XXsaveXX/ , all experiments data will be stored in a folder)

> The output of Example 2
>> ./Example_data/RFX5_TGGAGC30NGAT_4mer


 

### K-mer files to OneLine files (one sequence (kmers), one text file):
**Example 1. (ATAC-seq data for pre-train):** `python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/processed_ATAC_seq_ENCODE_p100_4mer --save_dir /tmp/csyuxu/processed_ATAC_seq_ENCODE_p100_4mer_pretrainDataFolder`  
**Example 2. (HT-SELEX data for fine-tune (single experimental data)):** `python utils.py --kmer_2_file True --kmer_dir ./Examples/Example_data/RFX5_TGGAGC30NGAT_4mer --save_dir ./Examples/Example_data/RFX5_TGGAGC30NGAT_4mer_finetuneDataFolder --level_sub_folder 2`  
**Example 3. (HT-SELEX data for fine-tune (Batch experimental data)):** `python utils.py --kmer_2_file True --kmer_dir /tmp/csyuxu/PRJEB3289_4mer --save_dir /tmp/csyuxu/PRJEB3289_4mer_finetuneDataFolder --level_sub_folder 2`

* --kmer_2_file : set `True` to enable this operation;
* --kmer_dir : the path of k-mer files
* --save_dir: the saving path
* --level_sub_folder: subfolder level 

> The output of Example 2
>> ./Example_data/RFX5_TGGAGC30NGAT_4mer_finetuneDataFolder/RFX5_TGGAGC30NGAT_4mer

***************
 
## TODO:
Add a function to accept sequences as inputs directly (merge pre-processing step)


***************
 
## Contact:
Mr. Yu Xu, email: csyuxu@comp.hkbu.edu.hk; allanxu20@gmail.com  
Dr. Eric Lu Zhang, email: ericluzhang@hkbu.edu.hk  

Thanks for your attention!
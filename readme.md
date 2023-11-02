# TRAFICA: 


## (A) Code environment  
1. Creat a new conda environment with the provided '.yaml' file.
```
conda env create -f environment.yaml
```
2. Activate the environment
```
conda activate TRAFICA
```


## (B) Quick start (with example data) 

* [Pre-training from scratch and Fine-tuning the pre-trained using the example data](./Src/readme.md)


## (C) The availability of the pre-trained moel, the fine-tuned adapters, and Data availability 

* [The pre-trained model and the fine-tuned adapters available at  HuggingFace](https://huggingface.co/Allanxu/TRAFICA/tree/main)
* [Processed data available at Zenodo](https://zenodo.org/doi/10.5281/zenodo.8248339)


***************
  
## (D) (TODO: sorting) Tools for k-mer vocab construction, sequence2kmer, and oneline files

### Interface introduction:  
`python utils.py --help`
   

### K-mer vocabulary construction:  
**Example:** `python utils.py --vocab_construct True --k 4 --save_dir ./vocab_k_mer`
* --vocab_construct : set `True` to enable this operation;
* --k : the hyperparameter of `k`-mer
* --save_dir: the saving path

 



***************
 
## (E) Contact:
Mr. Yu Xu, email: csyuxu@comp.hkbu.edu.hk; allanxu20@gmail.com  
Dr. Eric Lu Zhang, email: ericluzhang@hkbu.edu.hk  

Thanks for your attention!
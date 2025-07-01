<div style="display: flex; justify-content: center;">
  <img src="logo.png" alt="Logo" style="width:580px; margin-top: 20px;">
</div>

# TRAFICA: An Open Chromatin Language Model to Improve Transcription Factor Binding Affinity Prediction

***************
## (*) System requirement
1. OS: Linux
2. Nvidia GPU (CUDA support is need):  
> The pre-training stage of TRAFICA (base-level tokenization-> max number of tokens) took about 5.8 days on a single Nvidia A100 GPU card.
3. Python and other dependencies: [environment.yaml](environment.yaml)


***************
## (A) Installation: Conda environment  
1. Creat a new conda environment with the provided '.yaml' file.
```
conda env create -f environment.yaml
```
2. Activate the conda environment
```
conda activate TRAFICA
```


## (B) Quick start 
1. Loading the pre-trained model and tokenizer using HuggingFace Interface
``` python
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification 


# configuration dict for different TRAFICA versions
config_dict = {
  'TRAFICA (BPE-1)': {'model_path':'Allanxu/TRAFICA-BPE1',
                      'tokenizer_path':'Allanxu/TRAFICA-BPE1',
                      'tokenization':'BPE'
                      },
  'TRAFICA (BPE-2)': {'model_path':'Allanxu/TRAFICA-BPE2',
                    'tokenizer_path':'zhihan1996/DNABERT-2-117M',
                    'tokenization':'BPE'
                    },
  'TRAFICA (4-mer)': {'model_path':'Allanxu/TRAFICA-4_mer',
                    'tokenizer_path':'Allanxu/TRAFICA-4_mer',
                    'tokenization':'4_mer'
                    },
  'TRAFICA (5-mer)': {'model_path':'Allanxu/TRAFICA-5_mer',
                    'tokenizer_path':'Allanxu/TRAFICA-5_mer',
                    'tokenization':'5_mer'
                    },
  'TRAFICA (6-mer)': {'model_path':'Allanxu/TRAFICA-6_mer',
                    'tokenizer_path':'Allanxu/TRAFICA-6_mer',
                    'tokenization':'6_mer'
                    },
  'TRAFICA (base-level)': {'model_path':'Allanxu/TRAFICA-Base_level',
                    'tokenizer_path':'Allanxu/TRAFICA-Base_level',
                    'tokenization':'Base-level'
                    }
}

# tokenizer
Tokenizer = AutoTokenizer.from_pretrained(config_dict['TRAFICA (base-level)']['tokenizer_path'], trust_remote_code=True)    

# model
config = AutoConfig.from_pretrained(config_dict['TRAFICA (base-level)']['model_path'])
config.num_labels = 1 


model = AutoModelForSequenceClassification.from_pretrained(config_dict['TRAFICA (base-level)']['model_path'], config=config, trust_remote_code=True)
```

2. Loading the fine-tuned LoRA module and affinity predictor for specific TFs
``` python
from peft import PeftModel
import torch 

lora_path = '/<Path of fine-tuned LoRA>/Base-level/PRJEB3289/10000/ATF7_TGGGCG30NCGT' # example for TF ATF7

# LoRA and Affinity predictor
state_dict = torch.load(os.path.join(lora_path,"predict_head_weights.pth"), weights_only=True)  
model.classifier.load_state_dict( state_dict['PREDICT_HEAD'] )
model = PeftModel.from_pretrained(model, os.path.join(lora_path,"lora_adapter"))
```
**Fine-tuned TF LoRAs**  
[Available at HuggingFace (large size)](https://huggingface.co/Allanxu/Finetuned_TRAFICA_All)

3. Make prediction
``` python
from util.py import piece_sequences # Src/util.py    

# Input construction
sequences = ['CCAGAAGACAACTTGTAGAAATAAGCAAAA', 'ATTGCGCCCCAGCCCCACACCCACACGCAT']
tokens_batch = piece_sequences(sequences, config_dict['TRAFICA (base-level)']['tokenization'])   
# tokens_batch = ['C C A G A A G A C A A C T T G T A G A A A T A A G C A A A A', 'A T T G C G C C C C A G C C C C A C A C C C A C A C G C A T']
inputs = Tokenizer(tokens_batch, return_tensors="pt", padding=True)

# Prediction
with torch.no_grad():
    outputs = model(**inputs)
    
logit = outputs.logits 
print(f"Predicted relative affinities: {logit.flatten()}")
```


## (C) Experimental result repreduction & Fine-tuning on custom datasets
* Details of TRAFICA pre-training and LoRA fine-tuning [(Click here)](./Src/readme.md)



## (D) Data availability

<!-- * The weights of pre-trained model is available at [the HuggingFace repository](https://huggingface.co/Allanxu/TRAFICA/tree/main)
* The datasets used for TRAFICA pre-training/fine-tuning and evaluation are available at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.8248339) -->

* HT-SELEX Benchmark: [Zenodo](https://zenodo.org/records/15781226)


***************
 
## (*) Contact:
Mr. Yu Xu, Email: csyuxu@comp.hkbu.edu.hk; allanxu20@gmail.com  
Prof. Eric Lu Zhang, Email: ericluzhang@hkbu.edu.hk  


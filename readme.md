<div style="display: flex; justify-content: center;">
  <img src="logo.png" alt="Logo" style="width: 180px; margin-top: 20px;">
</div>

## TRAFICA: An Open Chromatin Language Model to Improve Transcription Factor Binding Affinity Prediction

*The logo of TRAFICA was generated by using the application of StableDiffusionXL in [Poe](https://poe.com/)*


***************
## (*) System requirement
1. OS: Linux
2. Nvidia GPU (CUDA support is need):  
> The pre-training of TRAFICA took over five days on a single Nvidia A100 GPU card. The fine-tuning of TRAFICA took about 30 minutes for an HT-SELEX dataset on the same hardware device.
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


## (B) Quick start (with example data) 

[TRAFICA pre-training and fine-tuning (Click here for details)](./Src/readme.md)
* Pre-training from scratch and Fine-tuning the pre-trained using the example data


## (C) The availability of the pre-trained moel and Data 

* The weights of pre-trained model is available at [the HuggingFace repository](https://huggingface.co/Allanxu/TRAFICA/tree/main)
* The datasets used for TRAFICA pre-training/fine-tuning and evaluation are available at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.8248339)



***************
 
## (D) Contact:
Mr. Yu Xu, email: csyuxu@comp.hkbu.edu.hk; allanxu20@gmail.com  
Dr. Eric Lu Zhang, email: ericluzhang@hkbu.edu.hk  


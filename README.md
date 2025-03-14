# Deep-Learning-for-Image-Denoising-A-Neural-Restoration-Framework
A survey on fundamental and extensive deep learning networks for image restoration and denoising. 

## Requirments
I have developed this project using 
> `python3 : Python 3.8.10`<br>
> `torch : 1.5.0+cu101`<br> `torchvison: 0.6.0+cu101`
>  
for getting the required librariese try to run the command:
> `pip install -r requirements.txt`

all the experiments have been done using a single **TeslaT4** on **Google Colab**. 

## Training
for training thre model try to run the `inference/main.ipynb` file 
**Notice:** 
> For configurating the project try using the `config.json` file.
> options on `config.json`:
* `model`: the architecture of the model.
  * `U-net` , `"VAE"`
* `save_weight_path`: the checkpoint folder for saving the model weigths in .pt format and the loss maps.
  * `null` ,`defined_path` 
* `load_weigth_path`: for loading the checkpoint path fro the model.
  * `null` ,`defined_path` 
* `datset`: defining the desired dataset for teh experiment.
  * `STL10` , `Cifar100` 
* `device` : the device used for training the model on:
  * `"cpu"` ,  `"cuda"`

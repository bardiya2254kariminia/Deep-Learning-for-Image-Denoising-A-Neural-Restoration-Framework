# Deep-Learning-for-Image-Denoising-A-Neural-Restoration-Framework
A survey on fundamental deep learning networks for image restoration and denoising. 

**Notice:**
> for configurating the project try using the `config.json` file.
> options on `config.json`:
* `model`: the architecture of the model.
  * `U-net` , `"VAE"`
* `save_weight_path`: the checkpoint path for saving the model weigths in .pt format and the loss maps.
  * `null` ,`defined_path` 
* `load_weigth_path`: for loading the checkpoint path fro the model.
  * `null` ,`defined_path` 
* `datset`: defining the desired dataset for teh experiment.
  * `STL10` , `Cifar100` 

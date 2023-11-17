# *Never Train from Scratch*: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors 

This repository provides implementations for experiments in the paper

The code and structure of this repository is based upon the official code of S4, please refer to
their readme file [README_S4.md](README_S4.md) for more information, most relevant are the 'Experiments', 
'Respository Structure' and 'README' sections.

## Requirements
It is recommended to use Python 3.10+ and Pytorch 2.0+  to allow for fast 
and memory-efficient implementation of attention functions, which should be installed via
the official Pytorch website: https://pytorch.org/get-started/locally/

Other packages are listed in [requirements.txt](./requirements.txt) and can be installed via:
~~~
pip install -r requirements.txt
~~~



## Experiments

Throughout all section pretraining and fine-tuning need to be executed separatly.\
config files for pretraining experiments have the postfix ```*-lm``` or ```*-mlm```.

### Section 3.1-3.5: Pretraining and Fine-tuning on LRA

Config file for LRA experiments can be found in ```configs/experiment/lra```

The following pretrains a transformer model with 50% masking ratio or causal masking on the Image task from LRA.
~~~
python train.py experiment=lra/transformer-lra-cifar-mlm dataset.mlm_prob=0.5 train.manual_checkpoints=true
python train.py experiment=lra/transformer-lra-cifar-mlm dataset.causal_lm=true dataset.mlm_prob=0.0 \
model.layer.causal=true train.manual_checkpoints=true
~~~
the ```train.manual_checkpoints=true``` flag keeps track of the last epoch and best model checkpoint for the desired metric, 
specified by ```train.monitor``` in the config file, automatic model checkpoints are generated via ```wandb```.

Pretrained models are saved in: 
~~~ 
outputs/<out_dir>/manual_checkpoints/version_<version>/last.ckpt
outputs/<out_dir>/manual_checkpoints/version_<version>/val/best.ckpt
~~~

For fine-tuning on the Image task from LRA from a pretraing checkpoint the following is used:
~~~
python train.py experiment=lra/transformer-lra-cifar train.pretrained_model_path=<path/to/pretrained_model.ckpt> 
~~~

When the ```train.pretrained_model_path``` flag is not specified the model is trained from scratch.

For sample complexity experiments (section 3.4) the additional flags ```+train.limit_train_samples=<sample_ratio>``` 
```+trainer.max_steps=<num_steps>``` are used to adjust the sample size and number of training steps. 
and number of training steps. For example:

~~~
python train.py experiment=lra/transformer-lra-cifar-mlm dataset.mlm_prob=0.5 train.manual_checkpoints=true 
+train.limit_train_samples=0.1 +trainer.max_steps=1000 hydra.run.dir=outputs/my_dir
~~~

Pretrains a transformer on the cifar with MLM objective, 50% masking, using 10% of the data and for 1000 gradient steps.
The model checkpoints are save to: 
~~~
outputs/my_dir/manual_checkpoints/version_<version>/last.ckpt
outputs/my_dir/manual_checkpoints/version_<version>/val/best.ckpt
~~~

For fine-tuning Pythia 70M (section 3.5) configs are provided in: ```configs/experiment/lra/llm```.

### Section 3.7: Additional Experiments

Using the same structure as experiments described in section 3.1-3.5, configs for additional experiments can 
can be found in:
~~~
BIDMC: configs/experiment/bidmc
Speech Commands: configs/experiment/sc
sCIFAR: configs/experiment/cifar
~~~


### Additional Notes on S4 Kernel installation and PyKeops
It is highly recommended to install the S4 kernel in this repository, provided by the original authors.
In other cases the default fallback uses the PyKeOps library, 
for which the cuda version of torch and pykeops should be verified and matching to avoid issues.
For additional issues with PyKeOps see:
https://www.kernel-operations.io/keops/python/installation.html#troubleshooting
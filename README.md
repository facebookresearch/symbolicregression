# Deep Symbolic Regression

This repository contains code for the paper [End-to-end symbolic regression with transformers](https://arxiv.org/abs/2204.10532).
An interactive demonstration of the paper may be found [here](https://symbolicregression.metademolab.com/).

The code is based on the repository [Deep Learning for Symbolic Mathematics](https://github.com/facebookresearch/SymbolicMathematics).
Most of the code specific to recurrent sequences lies in the folder ```src/envs```.

## Install dependencies 
Using conda and the environment.yml file:

```conda env create --name symbolic regression --file=environment.yml```

Also manually install a fork of sympytorch:

```pip install git+https://github.com/pakamienny/sympytorch```


## Run the model

To launch a model training use with additional arguments (arg1,val1), (arg2,val2):

```python train.py --arg1 val1 --arg2 --val2```

All hyper-parameters related to training are specified in parsers.py, and environment HPs are in envs/environment.py

To launch evaluation, please use the flag ```reload_checkpoint``` to specify in which folder the saved model is located.
```python evaluate.py --reload_checkpoint XXX```

## Try out a pre-trained model

We include a small notebook that loads a pre-trained model you can play with in ```Example.ipynb```

You can also check the demo website where you can play with the model without a single line of code [here](https://symbolicregression.metademolab.com/).

## Multinode training

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit) with grid-search:
```
pip install submitit
```

To launch a run on 2 nodes with 8 GPU each, use the ```submit.py``` script.

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [SymPy](https://www.sympy.org/)
- [PyTorch](http://pytorch.org/) (tested on version 1.3)

## Citation

If you want to reuse this material, please considering citing the following:
```
@article{kamienny2022end,
  title={End-to-end symbolic regression with transformers},
  author={Kamienny, Pierre-Alexandre and d'Ascoli, St{\'e}phane and Lample, Guillaume and Charton, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:2204.10532},
  year={2022}
}
```

## License

The majority of this repository is released under the Apache 2.0 license as found in the LICENSE file.

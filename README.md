# Tilo's fork of Breaching - A Framework for Attacks against Privacy in Federated Learning

Contains code I'm using summer 2024 to investigate the impact of changing the optimizer used in training on gradient attacks. 


install with:

```bash
git clone -b enable-multi https://github.com/TiloRC/breaching.git
cd breaching
pip install .
```

## additional dependencies 

### How to install KFAC

```bash
git clone -b compatible https://github.com/TiloRC/kfac-pytorch
cd kfac-pytorch
pip install .
```

### How to get SSIM metric working

first install kornia and pywavelets

```bash
pip install kornia pywavelets
```

then install pytorch_wavelets
```bash
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```


## Notes for myself for how to get everything working on chameleon instance

First install miniconda with bash script from https://docs.anaconda.com/miniconda/#quick-command-line-install

Then create a conda enviorment with the right version of python (https://stackoverflow.com/questions/56713744/how-to-create-conda-environment-with-specific-python-version)

IMPORTANT: install pytorch with https://pytorch.org/get-started/locally/ before following the above steps. 

Finally do the above steps for installing kfac, etc. 



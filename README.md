# Anomaly Detection by GANs

## Introduction
Here is a PyTorch code of anomaly detection by GANs. The overall GAN structure is taken from [eriklindernoren's code](https://github.com/eriklindernoren/PyTorch-GAN).
* (DONE)anogan: A PyTorch implementation of [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/pdf/1703.05921.pdf)
* (ToDo)f-anogan: [f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks](https://www.sciencedirect.com/science/article/pii/S1361841518302640?casa_token=fhezR4xnEe0AAAAA:mLwi2qXPcbjuoZBGw7pWnWRO4lWvUl1bvNnY3NtrCtITnIAuTfU-bjmSaU5z7Rvhp2HTmfGcEw)
* (ToDo)timegan: [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks.pdf)

## Datasets
1. MNIST: it is automatically downloaded by torchvision. 
2. energy: download [appliance energy prediction dataset](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) into the data/energy folder.

## Train
Train GAN(DCGAN) with normal dataset
1. MNIST: Run train.py --dataset mnist --train
2. energy: Run train.py --dataset energy --train --data_path ../data/energy/energydata_complete.csv --window_size 28 --skip_size 7

## Test
Projection test dataset with sparse noise onto the pretrained GAN manifolds.
1. MNIST: Run test.py --dataset mnist --add_noise
2. energy: Run test.py --dataset energy --add_noise --data_path ../data/energy/energydata_complete.csv --window_size 28 --skip_size 7

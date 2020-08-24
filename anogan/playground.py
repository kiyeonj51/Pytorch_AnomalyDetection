import pandas as pd
import argparse
import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from anogan.dataset import TimeSeriesDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--data_path", type=str, default="../data/energy/energydata_complete.csv", help="data path")
opt = parser.parse_args()
print(opt)

training_params = {"batch_size": opt.batch_size,
                   "shuffle": True,
                   "drop_last": False}


df = pd.read_csv(opt.data_path)
print("")

# Configure data loader
os.makedirs("../data/energy", exist_ok=True)
training_set = TimeSeriesDataset(data_path=opt.data_path,window_size=32, skip_size=8)
training_generator = DataLoader(training_set, **training_params)

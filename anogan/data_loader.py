import os
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from anogan.dataset import TimeSeriesDataset
from anogan.util import AddSparseNoise


def mnist(opt):
    path = '../data/' + opt.dataset
    os.makedirs(path, exist_ok=True)
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": False}

    trans = [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    if not opt.train:
        trans.append(AddSparseNoise(opt.noise_level))
    training_set = MNIST(
        root=path,
        train=opt.train,
        download=opt.download,
        transform=transforms.Compose(trans),
    )
    training_generator = DataLoader(training_set, **training_params)
    return training_generator


def energy(opt):
    path = '../data/' + opt.dataset
    os.makedirs(path, exist_ok=True)
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": False}
    training_set = TimeSeriesDataset(data_path=opt.data_path,
                                     window_size=opt.window_size,
                                     skip_size=opt.skip_size,
                                     noise_level=opt.noise_level,
                                     add_noise=opt.add_noise,
                                     train = opt.train)
    training_generator = DataLoader(training_set, **training_params)
    return training_generator

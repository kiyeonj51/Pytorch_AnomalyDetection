from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import pandas as pd
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, window_size, skip_size, noise_level=0.1, add_noise=False, train=True):
        super(TimeSeriesDataset, self).__init__()
        df = pd.read_csv(data_path)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        num_train = int(x_scaled.shape[0]*0.8)
        if train:
            df = pd.DataFrame(x_scaled[:num_train,:])
        else:
            df = pd.DataFrame(x_scaled[num_train:,:])
        num_slice = int((len(df) - window_size) / skip_size)
        slices = []
        for i in range(num_slice):
            slice = df[i*skip_size : i*skip_size + window_size].values
            if add_noise:
                slice[torch.rand(slice.shape) < noise_level] = 1.
            slices.append(slice[None,:,:])
        self.slices = slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        slice = self.slices[index]
        return slice, index

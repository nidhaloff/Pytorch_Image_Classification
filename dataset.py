from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split


class MNISTData(Dataset):
    def __init__(self):
        train_data = pd.read_csv("./data/train.csv", dtype=np.float)
        self.X_train, self.X_test = train_data.iloc[:, 1:].to_numpy() / 255
        self.y_train = train_data.iloc[:, 0].to_numpy()
        self.n = self.X_train.shape[1]
        self.X_train = torch.tensor(self.X_train, dtype=torch.float)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)

    def __getitem__(self, item):
        return self.X_train[item], self.y_train[item]

    def __len__(self):
        return self.X_train.shape[0]

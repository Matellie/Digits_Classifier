import torch
from torch.utils.data import Dataset

from sklearn import datasets

import numpy as np
import os

class MNISTDigits8x8(Dataset):
    """
    My best result: 97% accuracy

    model = base_model_code.SimpleNeuralNet(input_size=64, hidden_size=32, num_classes=10)
    loss = nn.CrossEntropyLoss

    batch_size = 1024
    nb_workers = 0
    learning_rate = 0.00001
    nb_epochs = 200000
    """
    def __init__(self):
        super(MNISTDigits8x8, self).__init__()

        self.name = 'MNISTDigits8x8'

        xy = datasets.load_digits()
        self.x = torch.from_numpy(xy.data.astype(np.float32))
        y = torch.from_numpy(xy.target).type(torch.LongTensor)
        self.y = torch.flatten(y)

        self.nb_samples = xy.data.shape[0]
        
        self.nb_features = xy.data.shape[1]
        self.nb_classes = 10

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nb_samples
    
class MNISTDigits28x28_train(Dataset):

    def __init__(self):
        super(MNISTDigits28x28_train, self).__init__()

        self.name = 'MNISTDigits28x28'

        xy = np.loadtxt(os.path.join("data", "mnist_train.csv"), delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        y = torch.from_numpy(xy[:, 0]).type(torch.LongTensor)
        self.y = torch.flatten(y)

        self.nb_samples = xy.shape[0]
        
        self.nb_features = xy.shape[1] - 1
        self.nb_classes = 10

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nb_samples
    
class MNISTDigits28x28_test(Dataset):

    def __init__(self):
        super(MNISTDigits28x28_test, self).__init__()

        self.name = 'MNISTDigits28x28_test'

        xy = np.loadtxt(os.path.join("data", "mnist_test.csv"), delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        y = torch.from_numpy(xy[:, 0]).type(torch.LongTensor)
        self.y = torch.flatten(y)

        self.nb_samples = xy.shape[0]
        
        self.nb_features = xy.shape[1] - 1
        self.nb_classes = 10

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nb_samples
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.utils
from torcheval.metrics.functional import binary_accuracy
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class ClassifierBinary(nn.Module):
    def __init__(self, inputSize:int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=inputSize, out_features=80, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(80),
            nn.Linear(in_features=80, out_features=50, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(in_features=50, out_features=30, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(30),
            nn.Linear(in_features=30, out_features=1, bias=True),
            #nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.5, 0.5)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.network(x).ravel()
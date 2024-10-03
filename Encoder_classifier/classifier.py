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
            nn.Linear(in_features=inputSize, out_features=256, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=1, bias=True),
            #nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.5, 0.5)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.network(x).ravel()
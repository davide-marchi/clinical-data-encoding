import torch
import torch.nn as nn
from torcheval.metrics.functional import r2_score

class IMEO(nn.Module):
    def __init__(self, inputSize:int, total_binary_columns: int = 0, embedding_dim: int = 10):
        '''
        inputSize: int the number of data columns + mask columns (should be an even number)
        total_binary_columns: int the number of binary columns in the data (binary data should be the first columns)
        embedding_dim: int the number of neurons in the embedding layer
        '''
        super().__init__()
        self.total_binary_columns = total_binary_columns
        self.inputSize = inputSize
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=inputSize, out_features=40, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=40, out_features=40, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=40, out_features=30, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=30, out_features=embedding_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=embedding_dim, out_features=30, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=30, out_features=40, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=40, out_features=40, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=40, out_features=40, bias=True),
            nn.LeakyReLU()
        )
        self.binaOut = nn.Linear(in_features=40, out_features=total_binary_columns, bias=True)
        self.binActivation = nn.Sigmoid()
        self.contOut = nn.Linear(in_features=40, out_features=inputSize//2 - total_binary_columns, bias=True)
        #initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.5, 0.5)
    
    def forward(self, x):
        x = self.linear_relu_stack(x)
        x1 = self.binActivation(self.binaOut(x))
        x2 = self.contOut(x)
        x = torch.cat((x1, x2), dim=1)
        return x
    
    def compute_r_squared(self, batch, binCol: int = 0):
        x = batch
        val, mask = torch.chunk(x, 2, dim=1)
        y_hat = torch.mul(self(x), mask)
        y = torch.mul(val, mask)
        if self.total_binary_columns != 0 or binCol != 0:
            binCol = self.total_binary_columns if binCol == 0 else binCol
            y_hat = y_hat[:, :-binCol]
            y = y[:, :-binCol]
        return r2_score(y, y_hat).item()
    
    def compute_accuracy(self, batch, binCol: int = 0):
        if self.total_binary_columns == 0 and binCol == 0:
            return 0
        x = batch
        binCol = self.total_binary_columns if binCol == 0 else binCol
        val, mask = torch.chunk(x, 2, dim=1)
        y_hat = torch.mul(self(x), mask)
        y = torch.mul(val, mask)
        y_hat_bin = y_hat[:, :binCol]
        y_bin = y[:, :binCol]
        y_hat_bin = torch.round(y_hat_bin)
        return (torch.sum(y_hat_bin == y_bin) / (y_hat_bin.shape[0] * binCol)).item()
    
    def training_step(self, batch, weightedLoss:bool = True):
        x = batch
        val, mask = torch.chunk(x, 2, dim=1)
        y_hat = torch.mul(self(x), mask)
        y = torch.mul(val, mask)
        if self.total_binary_columns == 0:
            return nn.functional.mse_loss(y, y_hat)
        y_hat_bin = y_hat[:, :self.total_binary_columns]
        y_bin = y[:, :self.total_binary_columns]
        loss_bin = 0
        for i in range(self.total_binary_columns):
            loss_bin += nn.functional.binary_cross_entropy(y_hat_bin[:, i], y_bin[:, i])
        y_hat_cont = y_hat[:, :-self.total_binary_columns]
        y_cont = y[:, :-self.total_binary_columns]
        loss_cont = nn.functional.mse_loss(y_cont, y_hat_cont)
        if not weightedLoss:
            return loss_bin + loss_cont
        binaryWeight = self.total_binary_columns / (self.inputSize//2)
        loss = binaryWeight*loss_bin + (1 - binaryWeight)*loss_cont
        return loss
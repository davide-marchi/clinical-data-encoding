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
            nn.Linear(in_features=inputSize, out_features=40, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(40),
            nn.Linear(in_features=40, out_features=40, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(40),
            nn.Linear(in_features=40, out_features=30, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(30),
            nn.Linear(in_features=30, out_features=1, bias=True),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.5, 0.5)
    
    def saveModel(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, x):
        return self.network(x)
    
    def binary_loss(self, output:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        output = output.squeeze()
        target = target.squeeze()
        return nn.functional.binary_cross_entropy(output, target)
        print(f'x: {x.shape}')
        print(f'y: {y.shape}')
        y_hat = self(x)
        return nn.functional.binary_cross_entropy(y_hat, y)
    
    def compute_accuracy(self, input:torch.Tensor, target:torch.Tensor) -> float:
        input = input.squeeze()
        target = target.squeeze()
        return binary_accuracy(input, target)
    
    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.5, 0.5)

    def fit(self, train_data, tr_out, val_data, val_out, optimizer, device, num_epochs, batch_size, print_every=10, plot=True, preprocess= lambda x:x):

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        print(f'output: {tr_out.shape}')
        print(f'input: {train_data.shape}')
        if train_data.shape[0] != tr_out.shape[0]:
            print("Error: The number of samples in the input and output tensors must match")
        print(f"Input: {val_data.shape}")
        print(f"Output: {val_out.shape}")
        if val_data.shape[0] != val_out.shape[0]:
            print("Error: The number of samples in the input and output tensors must match")
        train_loader = DataLoader(TensorDataset(train_data, tr_out), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data, val_out), batch_size=batch_size)

        self.to(device)
        self.train()

        for epoch in range(num_epochs):
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            epoch_train_accuracy = 0.0
            epoch_val_accuracy = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                x, y = batch
                x = preprocess(x)
                x = x.to(device)
                y = y.to(device)

                y_hat = self(x)
                loss = self.binary_loss(y_hat, y)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                epoch_train_accuracy += self.compute_accuracy(y_hat, y)

            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x = preprocess(x)
                    x = x.to(device)
                    y = y.to(device)

                    y_hat = self(x)
                    loss = self.binary_loss(y_hat, y)

                    epoch_val_loss += loss.item()
                    epoch_val_accuracy += self.compute_accuracy(y_hat, y)

            epoch_train_loss /= len(train_loader)
            epoch_val_loss /= len(val_loader)
            epoch_train_accuracy /= len(train_loader)
            epoch_val_accuracy /= len(val_loader)

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_accuracies.append(epoch_train_accuracy)
            val_accuracies.append(epoch_val_accuracy)

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")

        if plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='Train Accuracy')
            plt.plot(val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.show()

        return train_losses, val_losses, train_accuracies, val_accuracies
    
class DatasetClassificator(TensorDataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
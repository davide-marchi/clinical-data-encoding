import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
import torch.nn as nn
import torch.utils
from torcheval.metrics.functional import binary_accuracy


class BaseClassifier(nn.Module):
    """
    A PyTorch module defining a neural network architecture for classification.
    """
    
    def __init__(self, inputSize:int) -> None:
        """
        Constructor for the BaseClassifier class.

        Args:
            inputSize (int): The size of the input features.
        """
        super().__init__()
        # Define the neural network architecture using nn.Sequential
        self.network = nn.Sequential(
            nn.Linear(in_features=inputSize, out_features=110, bias=True),  # Input layer
            nn.LeakyReLU(),  # Activation function
            nn.BatchNorm1d(110),  # Batch normalization
            nn.Linear(in_features=110, out_features=65, bias=True),  # Hidden layer
            nn.LeakyReLU(),  # Activation function
            nn.BatchNorm1d(65),  # Batch normalization
            nn.Linear(in_features=65, out_features=23, bias=True),  # Hidden layer
            nn.LeakyReLU(),  # Activation function
            nn.BatchNorm1d(23),  # Batch normalization
            nn.Linear(in_features=23, out_features=1, bias=True),  # Output layer
            nn.Sigmoid()  # Activation function for binary classification
        )
        # Initialize weights and biases using Kaiming uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.5, 0.5)


    def forward(self, x):
        """
        Forward pass method of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.network(x)
    
    def binary_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the binary cross-entropy loss.

        Args:
            output (torch.Tensor): Predicted output tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Binary cross-entropy loss tensor.
        """
        # Squeeze the output and target tensors to remove any unnecessary dimensions
        output = output.squeeze()
        target = target.squeeze()

        # Compute the binary cross-entropy loss
        loss = nn.functional.binary_cross_entropy(output, target)

        return loss

    def train_model(self, train_loader, optimizer, device, num_epochs):
        """
        Train the model using the provided data.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            optimizer: Optimizer used for training.
            device (torch.device): Device to run the training on (GPU or CPU).
            num_epochs (int): Number of training epochs.

        Returns:
            list: List of training losses.
            list: List of training accuracies.
        """
        train_losses = []  # List to store training losses
        train_accuracies = []  # List to store training accuracies

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            
            epoch_train_loss = 0.0  # Initialize epoch training loss
            epoch_train_accuracy = 0.0  # Initialize epoch training accuracy

            # Iterate over the training DataLoader
            for batch in train_loader:
                x, y = batch
                x = x.to(device)  # Move input data to the appropriate device
                y = y.to(device)  # Move target data to the appropriate device

                optimizer.zero_grad()  # Zero the gradients

                y_hat = self.forward(x)  # Forward pass
                loss = self.binary_loss(y_hat, y)  # Compute the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the weights

                epoch_train_loss += loss.item()  # Accumulate the loss

            # Calculate the accuracy for the epoch
            epoch_train_accuracy = self.compute_accuracy(train_loader, device)

            # Calculate the average loss for the epoch
            avg_loss = epoch_train_loss / len(train_loader.dataset)    

            # Print the epoch number and average loss
            print(f'Epoch: {epoch}, Average Loss: {avg_loss}')

            # Append the average loss and accuracy to the respective lists
            train_losses.append(avg_loss)
            train_accuracies.append(epoch_train_accuracy)

        return train_losses, train_accuracies

    
    def compute_accuracy(self, data_loader, device):
        """
        Compute the model accuracy on the data provided by the DataLoader.

        Args:
            data_loader (DataLoader): The DataLoader providing the data.
            device (torch.device): The device to perform the computation on (GPU or CPU).

        Returns:
            float: The model accuracy.
        """
        self.eval()  # Set the model to evaluation mode

        # Concatenate predictions and targets from all batches
        all_y_hat = torch.cat([self.forward(x.to(device)).cpu() for x, _ in data_loader])
        all_y = torch.cat([y for _, y in data_loader])

        # Remove any unnecessary dimensions from predictions
        all_y_hat = all_y_hat.squeeze()

        # Calculate the accuracy using binary_accuracy function
        accuracy = binary_accuracy(all_y_hat, all_y)

        return accuracy


    def plot_loss(self, train_losses, val_losses=None, figsize=(8,6), print_every=1):
        """
        Plot training and validation losses.

        Args:
            train_losses (list): List of training losses.
            val_losses (list): List of validation losses (optional).
            figsize (tuple): Size of the figure.
            print_every (int): Interval for printing training progress.
        """
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_losses, label='Train Loss')
        if val_losses:
            plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.xticks(epochs[::print_every])
        plt.show()

    def plot_accuracy(self, train_accuracies, val_accuracies=None, figsize=(8,6), print_every=1):
        """
        Plot training and validation accuracies.

        Args:
            train_accuracies (list): List of training accuracies.
            figsize (tuple): Size of the figure.
            val_accuracies (list): List of validation accuracies (optional).
            print_every (int): Interval for printing training progress.
        """
        epochs = range(1, len(train_accuracies) + 1)
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        if val_accuracies:
            plt.plot(epochs, val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracies')
        plt.legend()
        plt.xticks(epochs[::print_every])
        plt.show()
    
    def reset(self):
        """
        Reset the weights and biases of linear layers using Kaiming uniform initialization.

        This method iterates through all modules of the model and resets the weights
        and biases of linear layers (nn.Linear) using Kaiming uniform initialization
        for weights and uniform initialization in the range [-0.5, 0.5] for biases.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)  # Initialize weights using Kaiming uniform initialization
                nn.init.uniform_(m.bias, -0.5, 0.5)  # Initialize biases uniformly in the range [-0.5, 0.5]


    def saveModel(self, path):
        """
        Save the model's state dictionary to a file.

        Args:
            path (str): The path where the model's state dictionary will be saved.
        """
        torch.save(self.state_dict(), path)

import torch
import torch.nn as nn
import pandas as pd

device = torch.device(  "cuda" if torch.cuda.is_available() 
                        else  "mps" if torch.backends.mps.is_available()
                        else "cpu"
                    )
print("Device: ", device)


file_path = './Datasets/data.csv'
data = pd.read_csv(file_path)
data_array = data.to_numpy()
data_array = data_array[:, 1:-1]
#standardize each column independently
data_array = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0))
#normalize each column independently
#data_array = (data_array - data_array.mean(axis=0)) / data_array.std(axis=0)
print(data_array.shape)
#convert to float32
data_array = data_array.astype('float32')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=data_array.shape[1], out_features=18, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=18, out_features=15, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=15, out_features=5, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=5, out_features=15, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=15, out_features=18, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=18, out_features=data_array.shape[1], bias=True)
        )
        #initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        return torch.cat([self.linear_relu_stack(x1), x2], dim=1)
    
    def training_step(self, batch):
        x = batch
        zeros = torch.zeros(x.size()).to(device)
        y_hat = self(torch.cat([x, zeros], dim=1))
        y = torch.cat([x, zeros], dim=1)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss
    
model = NeuralNetwork().to(device)
print(model)

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.95, nesterov=True)

# Define the data loader
batch_size = 512
data_loader = torch.utils.data.DataLoader(data_array, batch_size=batch_size, shuffle=True)

# Train the network
num_epochs = 90
losses = []
for epoch in range(num_epochs):
    for batch in data_loader:
        # Move batch to device
        batch = batch.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss = model.training_step(batch)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
    # Print the loss for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
    losses.append(loss.item())

#show plot of loss
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.show()
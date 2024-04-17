import random
import numpy as np
import utilsData
import torch
import torch.nn as nn

device = torch.device(  "cuda" if torch.cuda.is_available() 
                        else  "mps" if torch.backends.mps.is_available()
                        else "cpu"
                    )
#we have small amount of data, so we will use cpu (looks faster)
device = torch.device("cpu")
print("Device: ", device)


data_array = utilsData.load_data()
mask = utilsData.get_mask(data_array)
data_array = utilsData.standardize_data(data_array, mask)
data_array = data_array.to_numpy()
mask = mask.astype('float32')
data_array = data_array.astype('float32')
data_array = np.concatenate((data_array, mask), axis=1)
data_array = data_array[0:420*8]
data_array = torch.from_numpy(data_array)
data_array = data_array.to(device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=data_array.shape[1]//2, out_features=5, bias=True),
            nn.ReLU(),
            #nn.Linear(in_features=18, out_features=18, bias=True),
            #nn.ReLU(),
            #nn.Linear(in_features=18, out_features=5, bias=True),
            #nn.ReLU(),
            nn.Linear(in_features=5, out_features=18, bias=True),
            nn.ReLU(),
            #nn.Linear(in_features=18, out_features=18, bias=True),
            #nn.ReLU(),
            nn.Linear(in_features=18, out_features=data_array.shape[1]//2, bias=True)
        )
        #initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.linear_relu_stack(x)
    
    def training_step(self, batch):
        x = batch
        val, mask = torch.chunk(x, 2, dim=1)
        y_hat = torch.mul(self(val), mask)
        y = torch.mul(val, mask)
        #y_hat = self(val)
        loss = nn.functional.mse_loss(y, y_hat)
        #if random.randint(0,100) == 100:
        #    print(val[1],'\n\n', y_hat[1], loss)
        return loss
    
model = NeuralNetwork().to(device)
print(model)

# Define the optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005)

# Define the data loader
batch_size = 420
data_loader = torch.utils.data.DataLoader(data_array, batch_size=batch_size, shuffle=True)

# Train the network
num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    for batch in data_loader:
        # Move batch to device
        #batch = batch.to(device)
        
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
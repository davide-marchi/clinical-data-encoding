import numpy as np
import utilsData
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

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
#split the data into training and validation using sklern
train_data, val_data = train_test_split(data_array, test_size=0.2, random_state=42)
train_data = torch.from_numpy(train_data)
val_data = torch.from_numpy(val_data)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=data_array.shape[1], out_features=40, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=30, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=30, out_features=5, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=30, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=30, out_features=40, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=data_array.shape[1]//2, bias=True)
        )
        #initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.5, 0.5)
    
    def forward(self, x):
        return self.linear_relu_stack(x)
    
    def training_step(self, batch):
        x = batch
        val, mask = torch.chunk(x, 2, dim=1)
        y_hat = torch.mul(self(x), mask)
        y = torch.mul(val, mask)
        loss = nn.functional.mse_loss(y, y_hat)
        return loss
    
model = NeuralNetwork().to(device)
print(model)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), weight_decay=2e-5, lr=0.0004)

# Define the data loader
batch_size = 100

# Train the network
num_epochs = 2500
losses_tr = []
losses_val = []
train_data = train_data.to(device)
val_data = val_data.to(device)
data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        
    #compute the validation loss
    val_loss = model.training_step(val_data)
    losses_val.append(val_loss.item())
    loss_tr = model.training_step(train_data)
    losses_tr.append(loss_tr.item())
    if epoch % 500 == 0:
        print()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_tr.item():.4f}, Val Loss: {val_loss.item():.4f}", end='\r')
    if epoch == 0:
        print('')

print('Training complete\t\t\t\t')

#show plot of loss
import matplotlib.pyplot as plt
plt.plot(losses_tr, '-b')
plt.plot(losses_val, '-r')
plt.title('Loss vs. epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.show()
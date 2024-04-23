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
data_array, binary_clumns = utilsData.standardize_data(data_array, mask)
data_array = data_array.to_numpy()
mask = mask.astype('float32')
data_array = data_array.astype('float32')
data_array = np.concatenate((data_array, mask), axis=1)
#split the data into training and validation using sklern
train_data, val_data = train_test_split(data_array, test_size=0.2, random_state=42)
train_data = torch.from_numpy(train_data)
val_data = torch.from_numpy(val_data)

print(f'Number of binary columns: {binary_clumns}')

class NeuralNetwork(nn.Module):
    def __init__(self, total_binary_columns: int = 0, embedding_dim: int = 10):
        super().__init__()
        self.total_binary_columns = total_binary_columns
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=data_array.shape[1], out_features=40, bias=True),
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
        self.contOut = nn.Linear(in_features=40, out_features=data_array.shape[1]//2 - total_binary_columns, bias=True)
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
    
    def mse_loss(self, batch):
        x = batch
        val, mask = torch.chunk(x, 2, dim=1)
        y_hat = torch.mul(self(x), mask)
        y = torch.mul(val, mask)
        return nn.functional.mse_loss(y, y_hat)
    
    def compute_r_squared(self, batch):
        x = batch
        val, mask = torch.chunk(x, 2, dim=1)
        y_hat = torch.mul(self(x), mask)
        y = torch.mul(val, mask)
        if self.total_binary_columns == 0:
            y_mean = torch.mean(y)
            ss_tot = torch.sum((y - y_mean)**2)
            ss_res = torch.sum((y - y_hat)**2)
            return (1 - ss_res / ss_tot).item()
        else:
            y_hat_cont = y_hat[:, :-self.total_binary_columns]
            y_cont = y[:, :-self.total_binary_columns]
            y_mean = torch.mean(y_cont)
            ss_tot = torch.sum((y_cont - y_mean)**2)
            ss_res = torch.sum((y_cont - y_hat_cont)**2)
            return (1 - ss_res / ss_tot).item()
    
    def compute_accuracy(self, batch):
        if self.total_binary_columns == 0:
            return 0
        x = batch
        val, mask = torch.chunk(x, 2, dim=1)
        y_hat = torch.mul(self(x), mask)
        y = torch.mul(val, mask)
        y_hat_bin = y_hat[:, :self.total_binary_columns]
        y_bin = y[:, :self.total_binary_columns]
        y_hat_bin = torch.round(y_hat_bin)
        return (torch.sum(y_hat_bin == y_bin) / (y_hat_bin.shape[0] * self.total_binary_columns)).item()
    
    def training_step(self, batch):
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
        loss = loss_bin + loss_cont
        return loss

model = NeuralNetwork(total_binary_columns=binary_clumns, embedding_dim=5).to(device)
#print(model)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), weight_decay=2e-5, lr=0.0004)

# Define the data loader
batch_size = 100

# Train the network
num_epochs = 1000
losses_tr = []
losses_val = []
accuracy = [0]
r_squared = []
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
    mse_loss = model.mse_loss(val_data)
    accuracy.append(model.compute_accuracy(val_data))
    r_squared.append(model.compute_r_squared(val_data))
    losses_tr.append(loss_tr.item())
    if epoch % 200 == 0:
        print()
    print(f"Epoch [{epoch+1}/{num_epochs}], \
Loss: {loss_tr.item():.4f}, \
Val Loss: {val_loss.item():.4f}, \
Val MSE: {mse_loss.item():.4f}, \
Val R^2: {r_squared[-1]:.2f} \
Val acc: {accuracy[-1]:.2f}", end='\r')
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

plt.plot(r_squared, '-b')
plt.plot(accuracy, '-r')
plt.title('R^2 and Accuracy vs. epochs')
plt.xlabel('Epochs')
plt.ylabel('R^2 and Accuracy')
plt.legend(['R^2', 'Accuracy'])
plt.show()
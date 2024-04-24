from modelEncoderDecoderAdvancedV2 import IMEO
import numpy as np
import utilsData
import torch
import torch.nn as nn
from torcheval.metrics.functional import r2_score
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

model = IMEO(inputSize=data_array.shape[1], total_binary_columns=binary_clumns, embedding_dim=5)

model.to(device)


# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.5e-5, lr=0.0002)

# Define the data loader
batch_size = 100

# Train the network
num_epochs = 600
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
    accuracy.append(model.compute_accuracy(val_data, binary_clumns))
    r_squared.append(model.compute_r_squared(val_data, binary_clumns))
    losses_tr.append(loss_tr.item())
    if epoch % 200 == 0:
        print()
    print(f"Epoch [{epoch+1}/{num_epochs}], \
Loss: {loss_tr.item():.4f}, \
Val Loss: {val_loss.item():.4f}, \
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
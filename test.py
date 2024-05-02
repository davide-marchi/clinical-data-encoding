from modelEncoderDecoderAdvancedV2 import IMEO, trainIMEO
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
train_data, val_data = train_test_split(data_array, test_size=0.2, random_state=32)
train_data = torch.from_numpy(train_data)
val_data = torch.from_numpy(val_data)

binary_loss_weight = 14/18
batch_size = 100
learning_rate = 0.002
plot = True
print(f'Number of binary columns: {binary_clumns}')
print(f'Total number of columns: {data_array.shape[1]//2}')
print(f'Binary loss weight: {binary_loss_weight}')
print(f'Batch size: {batch_size}')
print(f'Learning rate: {learning_rate}')

model = IMEO(inputSize=data_array.shape[1], total_binary_columns=binary_clumns, embedding_dim=5)

model.to(device)


optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.5e-5, lr=learning_rate)

# Train the network
num_epochs = 300
losses_tr = []
losses_val = []
accuracy = [0]
r_squared = []
weighed_goodness = []
train_data = train_data.to(device)
val_data = val_data.to(device)
data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

history = trainIMEO(model, 
                    train_data, 
                    val_data, 
                    optimizer, 
                    binary_clumns, 
                    device, 
                    num_epochs, 
                    batch_size, 
                    binary_loss_weight, 
                    plot=False,
                    print_every=50
                )

exit()

if not plot: exit()

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
plt.plot(weighed_goodness, '-g')
plt.title('R^2 and Accuracy and a^2 vs. epochs')
plt.xlabel('Epochs')
plt.ylabel('R^2 and Accuracy and a^2')
plt.legend(['R^2', 'Accuracy', 'a^2'])
plt.show()
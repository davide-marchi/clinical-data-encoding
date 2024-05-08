from Encoder_classifier.modelEncoderDecoderAdvancedV2 import IMEO
from Encoder_classifier.classifierModelGenralFirstVersionBasic import ClassifierBinary
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
data_array = data_array.iloc[:, :-1]
mask = utilsData.get_mask(data_array)
data_array, binary_clumns = utilsData.standardize_data(data_array, mask)
data_array = data_array.to_numpy()
mask = mask.astype('float32')
data_array = data_array.astype('float32')
data_array = np.concatenate((data_array, mask), axis=1)
#split the data into training and validation using sklern
train_data, val_data = train_test_split(data_array, test_size=0.2, shuffle=False)
train_data = torch.from_numpy(train_data)
val_data = torch.from_numpy(val_data)

binary_loss_weight = 14/18
batch_size = 100
learning_rate = 0.002
plot = False
embedding_dim = 5
print(f'Number of binary columns: {binary_clumns}')
print(f'Total number of columns: {data_array.shape[1]//2}')
print(f'Binary loss weight: {binary_loss_weight:.3}')
print(f'Batch size: {batch_size}')
print(f'Learning rate: {learning_rate}')
print(f'Embedding dimension: {embedding_dim}')

encoder_decoder = IMEO(inputSize=data_array.shape[1], total_binary_columns=binary_clumns, embedding_dim=embedding_dim)

print(encoder_decoder)

encoder_decoder.to(device)


optimizer = torch.optim.Adam(encoder_decoder.parameters(), weight_decay=0.5e-5, lr=learning_rate)

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

history = encoder_decoder.fit(train_data,
                    val_data,
                    optimizer,
                    device=device,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    binary_loss_weight=binary_loss_weight,
                    print_every=num_epochs//5,
                    plot=plot
                )
encoder_decoder.freeze()
print('\n\nTraining classification model')
classifier = ClassifierBinary(inputSize=embedding_dim)
classifier.to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0002, weight_decay=0.00005)
num_epochs = 100

data_out = utilsData.load_data()
data_out = data_out.to_numpy()[:, -1]
data_out = data_out.astype('float32')
data_out = torch.from_numpy(data_out)
tr_out, val_out = train_test_split(data_out, test_size=0.2, shuffle=False)
print(tr_out.shape)
print(classifier)
history = classifier.fit(train_data,
                         tr_out,
                        val_data,
                        val_out,
                        optimizer,
                        device=device,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        print_every=num_epochs//5,
                        plot=True,
                        preprocess=encoder_decoder.encode
                    )
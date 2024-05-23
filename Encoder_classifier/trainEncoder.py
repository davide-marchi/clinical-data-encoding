from modelEncoderDecoderAdvancedV2 import IMEO
import torch
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import dataset_loader, load_data

device = torch.device(  "cuda" if torch.cuda.is_available() 
                        else  "mps" if torch.backends.mps.is_available()
                        else "cpu"
                    )
#we have small amount of data, so we will use cpu (looks faster)
device = torch.device("cuda")
print("Device: ", device)

folderName = './Datasets/Cleaned_Dataset/'
fileName = 'chl_dataset.csv'
dataset = load_data(folderName + fileName)

dict = dataset_loader(dataset, 0.1, 0.2, 42)
tr_data = dict['tr_data']
tr_out = dict['tr_out']
val_data = dict['val_data']
val_out = dict['val_out']
binary_clumns = dict['bin_col']


binary_loss_weight = binary_clumns/(tr_data.shape[1]//2)
batch_size = 300
learning_rate = 0.002
plot = True
embedding_dim = 17
weight_decay = 0.2e-5
num_epochs = 300
masked_percentage = 0.25

print(f'Number of binary columns: {binary_clumns}')
print(f'Total number of columns: {tr_data.shape[1]/2}')
print(f'Binary loss weight: {binary_loss_weight:.3}')
print(f'Batch size: {batch_size}')
print(f'Learning rate: {learning_rate}')
print(f'Embedding dimension: {embedding_dim}')
print(f'Weight decay: {weight_decay}')
print(f'Number of epochs: {num_epochs}')

encoder_decoder = IMEO(
    inputSize=tr_data.shape[1], 
    total_binary_columns=binary_clumns, 
    embedding_percentage=0.25
    )


#print(encoder_decoder)

encoder_decoder.to(device)


optimizer = torch.optim.Adam(encoder_decoder.parameters(), 
                            weight_decay=weight_decay, 
                            lr=learning_rate)

# Train the network
losses_tr = []
losses_val = []
accuracy = [0]
r_squared = []
weighed_goodness = []
train_data = tr_data.to(device)
val_data = val_data.to(device)
data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

patience = 10

history = encoder_decoder.fit(train_data,
                    val_data,
                    optimizer,
                    device=device,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    binary_loss_weight=binary_loss_weight,
                    print_every=num_epochs//5,
                    plot=plot,
                    masked_percentage = masked_percentage,
                    early_stopping=patience
                )
print('\n\nModel Trained\n\n')
print('Saving model...')
encoder_decoder.saveModel('./Encoder_classifier/encoder_decoder.pth')
print('Model saved')
from modelEncoderDecoderAdvancedV2 import IMEO
from classifier import ClassifierBinary
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
device = torch.device("cpu")
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

batch_size = 100
learning_rate = 0.002
plot = False
weight_decay = 0.8e-5
num_epochs = 20

print(f'Number of binary columns: {binary_clumns}')
print(f'Total number of columns: {tr_data.shape[1]/2}')
print(f'Learning rate: {learning_rate}')
print(f'Weight decay: {weight_decay}')
print(f'Number of epochs: {num_epochs}')

encoder_decoder:IMEO = torch.load('./Encoder_classifier/encoder_decoder.pth')

encoder_decoder.to(device)

# Train the network
train_data = tr_data.to(device)
val_data = val_data.to(device)
data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

embedding_dim = encoder_decoder.encoder[-2].out_features

classifier = ClassifierBinary(inputSize=embedding_dim)

patience = 10

classifier.to(device)
history = classifier.fit(train_data, 
                         tr_out, 
                         val_data, 
                         val_out, 
                         optimizer=torch.optim.Adam(classifier.parameters(), lr=learning_rate), 
                         device=device, 
                         num_epochs=num_epochs, 
                         batch_size=batch_size, 
                         preprocess=encoder_decoder.encode,
                         print_every=num_epochs//10,
                         early_stopping=patience
                         )

print('\n\nModel Trained\n\n')
print('Saving model...')
#encoder_decoder.saveModel('./Classifier/classifier.pth')
print('Model saved')
from classifier import ClassifierBinary
from modelEncoderDecoderAdvancedV2 import IMEO
from itertools import product
import torch
from utilsData import dataset_loader, load_data

# CLASSIFIER PARAMETERS
batch_size = [100]
learning_rate = [0.002]
plot = [False]
weight_decay = [0.8e-5]
num_epochs = [20]

# ENCODER PARAMETERS
binary_loss_weight = 0.5
batch_size = 100
learning_rate = 0.002
plot = True
embedding_dim_list = [17]
weight_decay = 0.2e-5
num_epochs = 300
masked_percentage_list = [0.2]

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

for embedding_dim, masked_percentage in product(embedding_dim_list, masked_percentage_list):
    # create encoder
    encoder = IMEO(
        inputSize=tr_data.shape[1], 
        total_binary_columns=binary_clumns, 
        embedding_dim=embedding_dim,
        neurons_num=[100, 80, 40]
        )
    # fit encoder
    encoder.fit(
        tr_data, 
        tr_out, 
        val_data, 
        val_out, 
        binary_loss_weight = binary_loss_weight, 
        batch_size = batch_size, 
        learning_rate = learning_rate, 
        plot = plot, 
        weight_decay = weight_decay, 
        num_epochs = num_epochs, 
        masked_percentage = masked_percentage
    )
    # create classifier
    classifier = ClassifierBinary(inputSize=embedding_dim)
    # fit classifier
    classifier.fit(
        tr_data, 
        tr_out, 
        val_data, 
        val_out, 
        optimizer=torch.optim.Adam(classifier.parameters(), lr=learning_rate), 
        device=device, 
        num_epochs=num_epochs, 
        batch_size=batch_size, 
        preprocess=encoder.encode,
        print_every=num_epochs//10,
    )
    encoder.saveModel(f'./Encoder_classifier/encoder_{embedding_dim}_{masked_percentage}.pth')
    classifier.saveModel(f'./Encoder_classifier/classifier_{embedding_dim}_{masked_percentage}.pth')
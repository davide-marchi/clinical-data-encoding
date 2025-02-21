import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '../../Encoder_classifier'))

from typing import Tuple
import torch
import numpy as np
from Encoder_classifier.modelEncoderDecoderAdvancedV2 import IMEO
from utilsData import dataset_loader, load_data # for loading data

def get_model(model_name:str)->IMEO:
    return (torch.load(f'./Encoder_classifier/gridResults/Models/{model_name}', weights_only=False))

def encode_with_model(model_name:str, tr_data, other_data) -> Tuple[np.array, np.array]:
    model = get_model(model_name)
    tr_data_enc = model.encode(tr_data).detach().numpy()
    other_data_enc = model.encode(other_data).detach().numpy()
    return tr_data_enc, other_data_enc

def get_dataset(years_to_death:int=8):
                                                            # _ O _
    # Load data with dataset_loader to avoid segmentation fault   \|/
    folderName = f'./Datasets/Cleaned_Dataset_{years_to_death}Y/chl_dataset_known.csv'
    dataset = load_data(folderName)
    return dataset_loader(dataset, 0.1, 0.2, 42)
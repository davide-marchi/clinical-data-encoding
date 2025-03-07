import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '../../Encoder_classifier'))

from typing import Tuple
import torch
import numpy as np
from Encoder_classifier.modelEncoderDecoderAdvancedV2 import IMEO
from utilsData import dataset_loader, load_data, unpack_encoder_name

def get_model(model_name:str)->IMEO:
    model_data = torch.load(f'./Encoder_classifier/gridResults/Models/{model_name}', weights_only=False)
    constructor_args = unpack_encoder_name(model_name)
    model = IMEO(inputSize=136, total_binary_columns=46, embedding_percentage=float(constructor_args['emb_perc']))  # Assuming IMEO has a default constructor
    model.load_state_dict(model_data)
    return model

def encode_with_model(model_name:str, tr_data, other_data) -> Tuple[np.array, np.array]:
    model = get_model(model_name)
    if not isinstance(model, IMEO):
        raise TypeError(f"Expected model of type IMEO, but got {type(model)}")
    tr_data_enc = model.encode(tr_data).detach().numpy()
    other_data_enc = model.encode(other_data).detach().numpy()
    return tr_data_enc, other_data_enc

def get_dataset(years_to_death:int=8):
                                                            # _ O _
    # Load data with dataset_loader to avoid segmentation fault   \|/
    folderName = f'./Datasets/Cleaned_Dataset_{years_to_death}Y/chl_dataset_known.csv'
    dataset = load_data(folderName)
    return dataset_loader(dataset, 0.1, 0.2, 42)
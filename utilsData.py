import json
import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def set_gpu()->torch.device:
    device = torch.device(  "cuda" if torch.cuda.is_available() 
                            else  "mps" if torch.backends.mps.is_available()
                            else "cpu"
                        )
    return device

def set_cpu()->torch.device:
    return torch.device("cpu")

def unpack_encoder_name(encoder_string:str)->dict:
    encoder_string = encoder_string.split('_')
    'encoder_{en_bin_loss_w}_{en_bs}_{en_lr}_{en_emb_perc}_{en_wd}_{en_num_ep}_{en_masked_perc}_{en_pt}.pth'
    encoder_string = encoder_string[1:]
    encoder_string[-1] = encoder_string[-1].split('.pth')[0]
    return {
        'binary_loss_weight': encoder_string[0],
        'batch_size': encoder_string[1],
        'lr': encoder_string[2],
        'emb_perc': encoder_string[3],
        'wd': encoder_string[4],
        'num_ep': encoder_string[5],
        'masked_perc': encoder_string[6],
        'pt': encoder_string[7]
    }

def unpack_classifier_name(classifier_string:str)->dict:
    classifier_string = classifier_string.split('_')
    'classifier_{cl_bs}_{cl_lr}_{cl_wd}_{cl_num_ep}_{cl_pt}_{cl_loss_w}.pth'
    classifier_string = classifier_string[1:]
    classifier_string[-1] = classifier_string[-1].split('.pth')[0]
    return {
        'batch_size': classifier_string[0],
        'lr': classifier_string[1],
        'wd': classifier_string[2],
        'num_ep': classifier_string[3],
        'pt': classifier_string[4],
        'loss_w': classifier_string[5]
    }

def load_data(path:str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def get_mask(data: pd.DataFrame) -> np.ndarray:
    mask = data.copy()
    vesselsMask = mask['Vessels'].replace(0, np.nan)
    creatininaMask = mask['Creatinina'].replace(0, np.nan)
    mask['Vessels'] = vesselsMask
    mask['Creatinina'] = creatininaMask
    mask = mask.isnull().astype(int)
    mask = 1 - mask
    return mask.to_numpy()

def standardize_data(data: pd.DataFrame, mask: np.ndarray) -> Tuple[pd.DataFrame,int]:
    print(mask.shape)
    print(data.shape)
    binayColums = 0
    for col_value, col_mask in zip(data.columns, mask.T):
        #print(f"col max: {data[col_value].max()}")
        if data[col_value].max() == 1 and data[col_value].min() == 0:
            #print(f'Column {col_value} is binary')
            binayColums += 1
            continue
        mean = data[col_value][col_mask == 1].mean()
        variance = data[col_value][col_mask == 1].var()
        data[col_value] = (data[col_value] - mean) / np.sqrt(variance)
    return data, binayColums

def normalize_data(tr: pd.DataFrame, val:pd.DataFrame, test:pd.DataFrame , tr_mask: np.ndarray, unlabledData: pd.DataFrame = None) -> Tuple[pd.DataFrame,int]:
    binaryColums = 0
    for col_value, col_mask in zip(tr.columns, tr_mask.T):
        if tr[col_value].max() == 1 and tr[col_value].min() == 0:
            binaryColums += 1
            continue
        min = tr[col_value][col_mask == 1].min()
        max = tr[col_value][col_mask == 1].max()
        tr[col_value] = (tr[col_value] - min) / (max - min)
        val[col_value] = (val[col_value] - min) / (max - min)
        test[col_value] = (test[col_value] - min) / (max - min)
        if unlabledData is not None:
            unlabledData[col_value] = (unlabledData[col_value] - min) / (max - min)
    return tr, val, test, unlabledData, binaryColums

def dataset_loader(data: pd.DataFrame, val_size:float, test_size:float, random_state:int, oversampling:bool = False, unlabledDataset: pd.DataFrame = None) -> 'dict[str,torch.Tensor]':
    '''
    in the returned dictionary:
    - tr_data: training data for the classifier
    - val_data: validation data for everyone
    - test_data: test data for everyone but to be used only at the end
    - tr_unlabled: training data for the encoder (valid only if unlabledDataset is not None)

    - tr_out: training output for the classifier
    - val_out: validation output for the classifier
    - test_out: test output for the classifier but to be used only at the end

    '''
    mask = data.copy()
    mask = mask.iloc[:, :-1]
    mask = mask.isnull().astype(int)
    mask = 1 - mask
    mask = mask.to_numpy()

    if unlabledDataset is not None:
        mask_unk = unlabledDataset.copy()
        mask_unk = mask_unk.isnull().astype(int)
        mask_unk = 1 - mask_unk
        mask_unk = mask_unk.to_numpy()

    train_out = data.iloc[:, -1]
    data = data.iloc[:, :-1]

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    train_out, test_out = train_test_split(train_out, test_size=test_size, random_state=random_state)
    train_mask, test_mask = train_test_split(mask, test_size=test_size, random_state=random_state)
    
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=random_state)
    train_out, val_out = train_test_split(train_out, test_size=val_size, random_state=random_state)
    train_mask, val_mask = train_test_split(train_mask, test_size=val_size, random_state=random_state)

    train_data, val_data, test_data, unlabledDataset, binary_clumns = normalize_data(train_data, val_data, test_data, train_mask, unlabledDataset)

    train_data[train_mask == 0] = 0
    val_data[val_mask == 0] = 0
    test_data[test_mask == 0] = 0
    if unlabledDataset is not None:
        unlabledDataset[mask_unk == 0] = 0
    
    train_data = np.concatenate((train_data, train_mask), axis=1)
    val_data = np.concatenate((val_data, val_mask), axis=1)
    test_data = np.concatenate((test_data, test_mask), axis=1)
    if unlabledDataset is not None:
        unlabledDataset = np.concatenate((unlabledDataset, mask_unk), axis=1)

    train_data = train_data.astype(np.float32)
    val_data = val_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    if unlabledDataset is not None:
        unlabledDataset = unlabledDataset.astype(np.float32)

    train_data = torch.from_numpy(train_data)
    val_data = torch.from_numpy(val_data)
    test_data = torch.from_numpy(test_data)
    if unlabledDataset is not None:
        unlabledDataset = torch.from_numpy(unlabledDataset)
        unlabledDataset = torch.cat((train_data, unlabledDataset), 0)

    train_out = torch.from_numpy(train_out.to_numpy()).float()
    val_out = torch.from_numpy(val_out.to_numpy()).float()
    test_out = torch.from_numpy(test_out.to_numpy()).float()
    
    return {'tr_data': train_data,
            'tr_out': train_out,
            'val_data': val_data, 
            'val_out': val_out, 
            'test_data': test_data, 
            'test_out': test_out, 
            'bin_col': binary_clumns,
            'tr_unlabled': unlabledDataset}

def dataset_loader_full(years:int):
    '''
    function written as a wrapper of dataset loader
    in order to make the code more readable
    args:
    - years: number of years to death
    '''
    folderName = f'./Datasets/Cleaned_Dataset_{years}Y/'
    fileName_kn = 'chl_dataset_known.csv'
    fileName_unk = 'chl_dataset_unknown.csv'
    dataset = load_data(folderName + fileName_kn)
    dataset_unk = load_data(folderName + fileName_unk)

    return dataset_loader(dataset, 0.1, 0.2, 42, oversampling=False, unlabledDataset=dataset_unk)

def load_past_results_and_models(old_results:bool=False)->Tuple[list,list,set]:
    '''
    function to load the past results and models
    args:
    - old_results: if True, the function will load the old_results.json file
        to use set to True for testing purposes
    '''
    results = []
    existing_models = []
    validated_models = set()
    if os.path.exists(f'./Encoder_classifier/gridResults/{"old_" if old_results else ""}results.json'):
        with open(f'./Encoder_classifier/gridResults/{"old_" if old_results else ""}results.json', 'r') as f:
            results = json.load(f)
    if os.path.exists('./Encoder_classifier/gridResults/Models/'):
        for file in os.listdir('./Encoder_classifier/gridResults/Models/'):
            if 'encoder' in file:
                existing_models.append(file)
    for elem in results:
        validated_models.add(elem['encoder_string'])
    return results, existing_models, validated_models
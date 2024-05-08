from typing import Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


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

def normalize_data(tr: pd.DataFrame, val:pd.DataFrame, test:pd.DataFrame , tr_mask: np.ndarray) -> Tuple[pd.DataFrame,int]:
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
    return tr, val, test, binaryColums

def dataset_loader(data: pd.DataFrame, val_size:float, test_size:float, random_state:int) -> dict:
    mask = data.copy()
    mask = mask.iloc[:, :-1]
    mask = mask.isnull().astype(int)
    mask = 1 - mask
    mask = mask.to_numpy()

    train_out = data.iloc[:, -1]
    data = data.iloc[:, :-1]

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    train_out, test_out = train_test_split(train_out, test_size=test_size, random_state=random_state)
    train_mask, test_mask = train_test_split(mask, test_size=test_size, random_state=random_state)
    
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=random_state)
    train_out, val_out = train_test_split(train_out, test_size=val_size, random_state=random_state)
    train_mask, val_mask = train_test_split(train_mask, test_size=val_size, random_state=random_state)

    train_data, val_data, test_data, binary_clumns = normalize_data(train_data, val_data, test_data, train_mask)

    train_data = np.concatenate((train_data, train_mask), axis=1)
    val_data = np.concatenate((val_data, val_mask), axis=1)
    test_data = np.concatenate((test_data, test_mask), axis=1)

    train_data = torch.from_numpy(train_data)
    val_data = torch.from_numpy(val_data)
    test_data = torch.from_numpy(test_data)
    return {'tr_data': train_data, 
            'tr_out': train_out, 
            'val_data': val_data, 
            'val_out': val_out, 
            'test_data': test_data, 
            'test_out': test_out, 
            'bin_col': binary_clumns}
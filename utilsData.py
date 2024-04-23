from typing import Tuple
import numpy as np
import pandas as pd


def load_data() -> pd.DataFrame:
    file_path = './Datasets/data_refactor.csv'
    data = pd.read_csv(file_path, sep=';')
    data = data.iloc[:, 1:-1]
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

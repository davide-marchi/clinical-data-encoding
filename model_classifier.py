import pandas as pd
from utilsData import get_mask, standardize_data, normalize_data, dataset_loader

file_path = './Datasets/Cleaned_Dataset/chl_dataset.csv'
data = pd.read_csv(file_path, sep=',')

print(data.head())
print(f'Train_data type: {type(data)}')

dataset_loader(data, 0.2, 0.2, 42)
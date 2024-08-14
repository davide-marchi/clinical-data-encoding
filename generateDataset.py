import argparse
import os

import numpy as np
import pandas as pd

# get the number passed as input
parser = argparse.ArgumentParser(description='Script to print an integer.')
parser.add_argument('-y', '--integer', type=int, help='An integer input', default=7)
args = parser.parse_args()
year_to_consider = args.integer

def reorder_columns(dataframe, columns):
    """
    Reorders the columns of the DataFrame based on the number of unique values in each column.
    Columns with only two unique values are placed first, followed by other columns.

    Parameters:
    - dataframe (DataFrame): The pandas DataFrame.
    - columns (list): A list containing the names of the columns to be reordered.

    Returns:
    - list: A list containing the reordered column names.
    """

    reordered_columns = []  # List to store column names with only two unique values
    non_bin_columns = []     # List to store column names with more than two unique values

    # Iterate through the specified columns
    for col in columns:
        # Check if the number of unique values in the column is equal to 2
        if dataframe[col].nunique() == 2:
            # If yes, append the column name to reordered_columns
            reordered_columns.append(col)
        else:
            # If no, append the column name to non_bin_columns
            non_bin_columns.append(col)

    # Combine the two lists to get the final reordered column order
    reordered_columns = reordered_columns + non_bin_columns
    
    return reordered_columns

def remove_outliers(dataframe, column_name, threshold, minor=False):
    """
    Remove outliers from a specific column in the dataframe based on a threshold.

    Args:
    - dataframe (DataFrame): The pandas DataFrame.
    - column_name (str): The name of the column containing the values to be checked for outliers.
    - threshold (float): The threshold value above which outliers will be removed.
    - minor (bool, optional): If True, remove values below the threshold. If False (default), remove values above the threshold.

    Returns:
    - DataFrame: The modified DataFrame with outliers removed.
    """
    if not minor:
        dataframe.loc[dataframe[column_name] > threshold, column_name] = np.nan
    else:
        dataframe.loc[dataframe[column_name] < threshold, column_name] = np.nan

    return dataframe[column_name]

# load the datasets
chl_dataset=pd.read_excel("Datasets/OrmoniTiroidei3Aprile2024.xlsx")
date_dataset=pd.read_excel("Datasets/DataPrelievo.xlsx")
creatinina_dataset=pd.read_excel("Datasets/Creatinina_AltriEsamiCorretti.xlsx")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

chl_dataset=chl_dataset.drop(columns=["HDL", "LDL", "Triglycerides", "Total cholesterol"])
chl_dataset=pd.merge(chl_dataset, creatinina_dataset, on="Number")
chl_dataset=pd.merge(chl_dataset, date_dataset, on=["Number"])
chl_dataset=chl_dataset.dropna(subset=["Number"])
chl_dataset=chl_dataset.drop(columns=["PCI", "Ictus", "Non Fatal AMI (Follow-Up)", "CABG ", 
                            "Suicide","Accident", "UnKnown", "Fatal MI or Sudden death", 
                            "Total mortality", "Collected by", "Cause of death", "Number"])

data_columns=chl_dataset.columns
reordered_columns=reorder_columns(chl_dataset, data_columns)
chl_dataset = chl_dataset.reindex(columns=reordered_columns)
chl_dataset=chl_dataset.drop(columns=["CardiopatiaCongenita"])

# cleaning from outliers
if True:
    chl_dataset["Glycemia"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="Glycemia",
                                            threshold=350)
    chl_dataset["Glycemia"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="Glycemia",
                                            threshold=2,
                                            minor=True)
    chl_dataset["TSH"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="TSH",
                                            threshold=21)
    chl_dataset["fT3"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="fT3",
                                            threshold=13)
    chl_dataset["fT4"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="fT4",
                                            threshold=40)
    chl_dataset["fT4"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="fT4",
                                            threshold=0.5,
                                            minor=True)
    chl_dataset["Vessels"]=remove_outliers(dataframe=chl_dataset,   #HMMMMM
                                            column_name="Vessels",  #HMMMMM
                                            threshold=0.5,          #HMMMMM
                                            minor=True)             #HMMMMM
    chl_dataset["HR"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="HR",
                                            threshold=190)
    chl_dataset["BMI"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="BMI",
                                            threshold=55)
    chl_dataset["Diastolic blood pressure"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="Diastolic blood pressure",
                                            threshold=141)
    chl_dataset["DimSettoIV"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="DimSettoIV",
                                            threshold=24)
    chl_dataset["DimPP"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="DimPP",
                                            threshold=20)
    chl_dataset["vsx"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="vsx",
                                            threshold=89)
    chl_dataset["Total cholesterol"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="Total cholesterol",
                                            threshold=420)
    chl_dataset["HDL"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="HDL",
                                            threshold=110)
    chl_dataset["LDL"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="LDL",
                                            threshold=300)
    chl_dataset["Triglycerides"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="Triglycerides",
                                            threshold=600)
    chl_dataset["Creatinina"]=remove_outliers(dataframe=chl_dataset,
                                            column_name="Creatinina",
                                            threshold=5)

combined_df = pd.DataFrame({
    'Follow Up Data': pd.to_datetime(chl_dataset['Follow Up Data']),
    'Data prelievo': pd.to_datetime(date_dataset['Data prelievo']),
    'Data of death': pd.to_datetime(chl_dataset['Data of death']),
    "CVD Death": pd.to_numeric(chl_dataset["CVD Death"])
})

combined_df['Death Difference'] = (combined_df['Data of death'] - combined_df['Data prelievo']).dt.days / 365.25

def set_target_correct(row, num_years=7):
    '''
    Set the target variable based on the patient's status within a specified number of years.

    Args:
    - row (pd.Series): A row of the DataFrame.
    - num_years (int, optional): The number of years to consider for the target variable. Defaults to 7.

    Returns:
    - str: The target variable value ('Alive', 'Deceased', 'CVD Deceased', or 'Unknown').
    '''

    # if the data of second visit is not available, the patient is considered unknown
    # if the the second visit is within num_years but the patient is alive, the patient is considered unknown
    # if the patient is dead (before num_years), the patient is considered dead if the cause of death is CVD
    # if the patient is alive, the patient is considered alive
    
    # non è stata effettuata la seconda visita
    if pd.isna(row['Follow Up Data']):
        return 'Unknown'
    # if the follow up data is within num_years
    if row['Follow Up Data'] - row['Data prelievo'] < pd.Timedelta(days=num_years*365.25):
        # the patient is dead before num_years
        if row['CVD Death'] == 1:
            return 'CVD Deceased'
        # the patient is either alive or dead due to other causes
        # if dead due to other causes we cannot know if the patient would have died due to CVD
        # if he is alive we cannot know if he will die due to CVD
        return 'Unknown'
    # if the follow up data is after num_years
    else:
        # the patient is alive (or it survived for more than num_years)
        return 'Alive'
        
            
apply_target = lambda row: set_target_correct(row, num_years=year_to_consider)
chl_dataset['Target'] = chl_dataset.apply(apply_target, axis=1)
chl_dataset=chl_dataset.drop(columns="CVD Death")
chl_dataset=chl_dataset.drop(columns="Data of death")
chl_dataset=chl_dataset.drop(columns="Follow Up Data")
chl_dataset=chl_dataset.drop(columns="Data prelievo")


# Get the directory of the script
script_directory = os.getcwd()
# Specify the folder name
folder_name = f'Datasets/Cleaned_Dataset_{year_to_consider}Y'
# Combine the script directory and folder name to get the full path
folder_path = os.path.join(script_directory, folder_name)
# Check if the folder exists, and create it if not
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# make 2 distinct datasets one with only unknown target rows and one with the others
chl_dataset_unknown = chl_dataset[chl_dataset['Target'] == 'Unknown']
chl_dataset_unknown = chl_dataset_unknown.drop(columns='Target')
chl_dataset_known = chl_dataset[chl_dataset['Target'] != 'Unknown']
chl_dataset_known['Target'] = chl_dataset_known['Target'].apply(lambda x: 0 if x == 'CVD Deceased' else 1)
# Save the datasets to CSV files
chl_dataset_unknown.to_csv(os.path.join(folder_path, 'chl_dataset_unknown.csv'), index=False)
chl_dataset_known.to_csv(os.path.join(folder_path, 'chl_dataset_known.csv'), index=False)
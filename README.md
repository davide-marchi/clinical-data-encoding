# clinical-data-encoding

Welcome to the Clinical Data Encoding project repository! This project focuses on encoding clinical data to create embeddings of subjects' health data using autoencoders. Additionally, we extend the main dataset with additional data from other datasets related to the same patients.

<div align="center">
<img hight="250" width="400" alt="GIF" align="center" src="https://github.com/davide-marchi/clinical-data-encoding/blob/main/Figures/homer-simpson-fat.gif">
</div>

</br>
</br>

## About
The primary objective of this project is to develop an efficient method for representing complex clinical data in a lower-dimensional space. By leveraging autoencoders, we aim to generate embeddings that capture the underlying structure of the data while preserving important information about the patients' health status. Our ultimate goal is to handle datasets with missing data and learn to perform imputation, creating consistent embeddings even for patients with missing data.

## Dataset
The main dataset used in this project is `OrmoniTiroidei3Aprile2024.xlsx`, which contains real clinical data related to thyroid disorders. Additionally, we augment the main dataset with additional data from other datasets pertaining to the same patients, enhancing the richness and diversity of the data.

## Methodology
We utilize an autoencoder architecture to encode the clinical data into a lower-dimensional space. The autoencoder comprises an encoder network that compresses the input data into a latent space representation and a decoder network that reconstructs the original input from the encoded representation. By training the autoencoder on the augmented dataset, we aim to learn meaningful embeddings that capture the essence of the patients' health data. We focus on handling missing data and aim to learn imputation techniques to create consistent embeddings for patients with missing data.

## Evaluation
To evaluate the effectiveness of our embeddings, we employ them as features to train a neural network for a classification task. The targets for the classification task are specified within the `Cleaning_Data.ipynb` notebook, where we also perform data cleaning and preprocessing.

## Authors
- Angelo Nardone
- Davide Borghini
- Davide Marchi
- Giordano Scerra

## Getting Started
To get started with the project, follow these steps:
1. Clone the repository to your local machine.
2. Install the necessary dependencies listed in the `requirements.txt` file.
3. Explore the codebase and experiment with different configurations and parameters.
4. Run the provided scripts to train the autoencoder on the augmented dataset and generate embeddings.
5. Use the embeddings as features to train a neural network for the classification task specified in the `Cleaning_Data.ipynb` notebook.

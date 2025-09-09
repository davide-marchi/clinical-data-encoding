# MIEO - Masked Input Encoded Output

<p align="center">
  <img src="Figures/MIEO Logo.png" alt="MIEO" height="110">
</p>

<p align="center">
  <a href="https://github.com/davide-marchi/clinical-data-encoding/stargazers">
    <img src="https://img.shields.io/github/stars/davide-marchi/clinical-data-encoding" alt="GitHub Stars">
  </a>
  <a href="https://github.com/davide-marchi/clinical-data-encoding/issues">
    <img src="https://img.shields.io/github/issues/davide-marchi/clinical-data-encoding" alt="GitHub Issues">
  </a>
  <a href="https://github.com/davide-marchi/clinical-data-encoding/pulls">
    <img src="https://img.shields.io/github/issues-pr/davide-marchi/clinical-data-encoding" alt="Pull Requests">
  </a>
  <a href="https://github.com/davide-marchi/clinical-data-encoding/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/davide-marchi/clinical-data-encoding" alt="Contributors">
  </a>
  <a href="https://github.com/davide-marchi/clinical-data-encoding">
    <img src="https://img.shields.io/github/repo-size/davide-marchi/clinical-data-encoding" alt="Repository Size">
  </a>
  <a href="https://github.com/davide-marchi/clinical-data-encoding">
    <img src="https://img.shields.io/github/last-commit/davide-marchi/clinical-data-encoding" alt="Last Commit">
  </a>
  <a href="https://github.com/davide-marchi/clinical-data-encoding/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/davide-marchi/clinical-data-encoding" alt="License">
  </a>
</p>

Self-supervised autoencoder for structured clinical data that handles missing values via input masking and a type-aware loss. Embeddings feed a downstream ANN to predict **cardiovascular death within 8 years** on an **IHD** cohort.  

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

## Modified library
To avoid 
    .venv/lib/python3.8/site-packages/skorch/net.py:2231: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`.

    We changed on skorch:
        From:   load_kwargs = {'map_location': map_location}
        To:     load_kwargs = {'map_location': map_location, 'weights_only':True}
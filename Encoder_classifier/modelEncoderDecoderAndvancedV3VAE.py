import torch
import torch.nn as nn
from torcheval.metrics.functional import r2_score

class MIEOVAE(nn.Module):
	def __init__(self, latent_dim:int=32, input_dim:int=100, hidden_dims:list=[128, 64], binary:int=None):
		'''
		MIOE VAE model
		A variational autoencoder that handles continuous and binary data
		Args:
			latent_dim (int): Dimension of the latent space
			input_dim (int): Number of input features
			hidden_dims (list): List of hidden dimensions for the encoder (the decoder is symmetric)
			binary (int): Number of binary featueres at the end of the input (if any)
		'''
		super().__init__()

		# --- Encoder ---
		self.encoder = nn.Sequential(
			nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
			nn.Tanh(),
		)
		for i in range(len(hidden_dims) - 1):
			self.encoder.append(nn.Linear)
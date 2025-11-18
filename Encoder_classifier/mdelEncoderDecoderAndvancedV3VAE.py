import torch
import torch.nn as nn
from torcheval.metrics.functional import r2_score

class MIEOVAE(nn.Module):
	def __init__(self, latent_dim:int=32, input_dim:int=100, hidden_dims:list=[128, 64], binary:int=None) -> None:
		'''
		MIOE VAE model
		A variational autoencoder that handles continuous and binary data
		Args:
			latent_dim (int): Dimension of the latent space
			input_dim (int): Number of input features
			
		'''
		super().__init__()

		# --- Encoder ---
		self.encoder = nn.Sequential(

		)
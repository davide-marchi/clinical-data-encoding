import torch
import torch.nn as nn
#from torcheval.metrics.functional import r2_score

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
			self.encoder.append(nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1]))
			self.encoder.append(nn.Tanh())
		
		# --- mu - logvar ---
		self.mu = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)
		self.logvar = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)

		# --- Decoder ---
		self.decoder = nn.Sequential(
			nn.Linear(in_features=latent_dim, out_features=hidden_dims[-1]),
			nn.Tanh(),
		)
		for i in range(len(hidden_dims) - 1):
			self.decoder.append(nn.Linear(in_features=hidden_dims[-i - 1], out_features=hidden_dims[-i -2]))
			if i != len(hidden_dims) -2:
				self.decoder.append(nn.Tanh())
		
	def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std
	
	def encode(self, x:torch.Tensor) -> tuple:
		encoded = self.encoder(x)
		mu = self.mu(encoded)
		logvar = self.logvar(encoded)
		return mu, logvar

	def forward(self, x:torch.Tensor) -> tuple:
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		reconstructed = self.decoder(z)
		return reconstructed, mu, logvar
	
	def kl(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
		# KL divergence between N(mu, var) and N(0, 1)
		return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()

	def get_loss(self, x:torch.Tensor, beta:float=1.0, null_mask:torch.Tensor=None) -> tuple:
		reconstructed, mu, logvar = self.forward(x)
		reconstruction_loss = nn.functional.mse_loss(reconstructed*null_mask, x*null_mask, reduction='mean')
		kl_loss = self.kl(mu, logvar)
		total_loss = reconstruction_loss + beta * kl_loss
		return total_loss, reconstruction_loss, kl_loss
	
	def get_loss_masked(self, x:torch.Tensor, mask_percentage:float=0.1, beta:float=1.0, null_mask:torch.Tensor=None) -> tuple:
		# Create a mask
		mask = torch.rand_like(x) > mask_percentage
		x_masked = x * mask.float()
		x_masked = x_masked - (~mask).float()
		#x_masked.requires_grad = False

		reconstructed, mu, logvar = self.forward(x_masked)
		reconstruction_loss = nn.functional.mse_loss(reconstructed*null_mask, x*null_mask, reduction='mean')
		kl_loss = self.kl(mu, logvar)
		total_loss = reconstruction_loss + beta * kl_loss
		return total_loss, reconstruction_loss, kl_loss
		

if __name__ == "__main__":
	model = MIEOVAE(hidden_dims=[60, 40, 30, 20, 3])
	print(model)
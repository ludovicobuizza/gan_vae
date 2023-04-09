import torch
import torch.nn as nn
from utils import make_module, make_final_decoder_layer


class VAE(nn.Module):
    def __init__(self, encoder_hyper_params, decoder_hyper_params):
        """Variational Autoencoder (VAE) model

        Args:
            encoder_hyper_params (dict): Dictionary of hyperparams for encoder.
                It should contain the following keys:
                - latent_dims (int): Dimensionality of the latent space.
                - hidden_channels (list): List of hidden dimensions.
                - kernels (list): List of kernel sizes.
                - strides (list): List of strides.
                - paddings (list): List of paddings.
                - in_channels (int): Number of input channels.
                - fc_neurons (int): Number of neurons in the fully connected
                  layer.
            decoder_hyper_params (dict): Dictionary of hyperparams for decoder.
                It should contain the following keys:
                - in_channels (int): Number of input channels.
                - hidden_channels (list): List of hidden channels.
                - kernels (list): List of kernel sizes.
                - strides (list): List of strides.
                - paddings (list): List of paddings.
                - out_channels (int): Number of output channels.
                - final_kernel (int): Kernel size of the final layer.


        """
        super(VAE, self).__init__()
        self.latent_dims = encoder_hyper_params["latent_dims"]
        self.encoder_hyper_params = encoder_hyper_params
        self.decoder_hyper_params = decoder_hyper_params

        self.encoder = self.make_module(conv_layer=nn.Conv2d,
                                         hyper_params=encoder_hyper_params,
                                         activation=nn.LeakyReLU)
        self.fc_mu = nn.Linear(encoder_hyper_params["fc_neurons"],
                               self.latent_dims)
        self.fc_var = nn.Linear(encoder_hyper_params["fc_neurons"],
                                self.latent_dims)

        last_dim = encoder_hyper_params["hidden_channels"][-1]
        self.decoder_input = nn.Linear(self.latent_dims, last_dim * 4)
        self.decoder = make_module(conv_layer=nn.ConvTranspose2d,
                                         hyper_params=decoder_hyper_params,
                                         activation=nn.LeakyReLU)
        self.final_layer = make_final_decoder_layer(decoder_hyper_params)

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, int(result.shape[1] / 4), 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        return self.decode(z), x, mu, log_var

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, generator_hyper_params: dict):
        """Initialise generator.

        Args:
            generator_hyper_params: dictionary with generator hyperparameters
        """
        super(Generator, self).__init__()
        gen_fm = generator_hyper_params["gen_feature_map"]
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                generator_hyper_params["latent_vector_size"],
                gen_fm * 8,
                4,
                1,
                0,
                bias=False,
            ),
            nn.BatchNorm2d(gen_fm * 8),
            nn.ReLU(True),
            # state size is (gen_fm * 8) x 4 x 4
            nn.ConvTranspose2d(gen_fm * 8, gen_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_fm * 4),
            nn.ReLU(True),
            # state size is (gen_fm * 4) x 8 x 8
            nn.ConvTranspose2d(gen_fm * 4, gen_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_fm * 2),
            nn.ReLU(True),
            # state size is (gen_fm * 2) x 16 x 16
            nn.ConvTranspose2d(gen_fm * 2, gen_fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_fm),
            nn.ReLU(True),
            # state size is gen_fm x 32 x 32
            nn.ConvTranspose2d(gen_fm, 3, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size is 3 x 32 x 32
        )

    def forward(self, z: torch.Tensor):
        """Forward pass through the generator."""
        out = self.conv(z)
        return out


class Discriminator(nn.Module):
    def __init__(self, discriminator_hyper_params: dict):
        """Initialise discriminator.

        Args:
            discriminator_hyper_params: dictionary with discriminator
                                        hyperparameters
        """
        super(Discriminator, self).__init__()
        disc_fm = discriminator_hyper_params["disc_feature_map"]
        self.conv = nn.Sequential(
            # input is 3 x 32 x 32
            nn.Conv2d(3, disc_fm, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is disc_fm x 32 x 32
            nn.Conv2d(disc_fm, disc_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_fm * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (disc_fim * 2) x 16 x 16
            nn.Conv2d(disc_fm * 2, disc_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_fm * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (disc_fm * 4) x 8 x 8
            nn.Conv2d(disc_fm * 4, disc_fm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_fm * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (disc_fm * 8) x 4 x 4
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(disc_fm * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass through the discriminator."""
        out = self.conv(x)
        out = self.final_layer(out)
        return out



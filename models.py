import torch
import torch.nn as nn
import numpy as np


def encoder_layers(n: int):
    standard_set = {'kernel_size': 3,
                    'stride': 2,
                    'padding': 1}

    layers = [nn.Conv2d(3, 8, **standard_set),
              nn.BatchNorm2d(8),
              nn.LeakyReLU(),
              nn.Dropout2d(p=0.25)]

    channel = 8

    for _ in range(int(np.log2(n) - 4)):
        layers.append(nn.Conv2d(channel, channel * 2, **standard_set))
        channel = channel * 2
        layers.append(nn.BatchNorm2d(channel))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout2d(p=0.25))

    return layers


class Encoder(nn.Module):

    def __init__(self, n: int, device):
        super(Encoder, self).__init__()

        self.device = device

        self.layers = encoder_layers(n)

        self.main = nn.Sequential(*self.layers)

        self.x = np.log2(n) - 4

        self.mu = nn.Linear(int(2 ** (self.x + 9)), 200)
        self.log_var = nn.Linear(int(2 ** (self.x + 9)), 200)

    def sampling(self, mu, log_var):
        epsilon = torch.randn_like(mu)

        epsilon = epsilon.to(self.device)

        out = mu + torch.exp(log_var / 2) * epsilon
        return out

    def forward(self, img):
        img = self.main(img)

        img = img.view(-1, int(2 ** (self.x + 9)))

        mu = self.mu(img)
        log_var = self.log_var(img)

        img = self.sampling(mu, log_var)
        return img.float()


def decoder_layers(n: int, activation=nn.LeakyReLU, batch_norm=True, dropout=True):
    standard_set = {'kernel_size': 4,
                    'stride': 2,
                    'padding': 1}

    layers = []

    num_layers = int(np.log2(n) - 3)
    for _ in range(num_layers):
        layers.append(nn.ConvTranspose2d(int(n), n // 2, **standard_set))
        n = n // 2
        layers.append(nn.BatchNorm2d(n)) if batch_norm else None
        layers.append(activation())
        layers.append(nn.Dropout2d(p=0.25)) if dropout else None

    return layers


class Decoder(nn.Module):

    def __init__(self, n: int):
        super(Decoder, self).__init__()

        self.x = np.log2(n) - 4

        self.first = nn.Linear(200, int(2 ** (9 + self.x)))

        self.main = nn.Sequential(*decoder_layers(2 ** (3 + self.x)))

        self.final = nn.Sequential(nn.ConvTranspose2d(8, 3,
                                                      kernel_size=4,
                                                      stride=2,
                                                      padding=1),
                                   nn.Sigmoid())

    def forward(self, vector):
        vector = self.first(vector)
        vector = vector.view(-1, int(2 ** (3 + self.x)), 8, 8)

        img = self.main(vector)
        img = self.final(img)
        return img


class VAE(nn.Module):

    def __init__(self, n, device):
        super(VAE, self).__init__()

        self.device = device

        self.encoder = Encoder(n, self.device)
        self.decoder = Decoder(n)

        self.epoch = 0

    def forward(self, img):
        img = self.encoder(img)
        img = self.decoder(img)

        return img


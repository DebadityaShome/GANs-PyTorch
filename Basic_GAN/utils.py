import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor, num_images=5, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Params:
        n_samples: scalar representing number of samples to generate
        z_dim: scalar representing dimension of noise vector
        device: the device type
    '''
    return torch.randn((n_samples, z_dim), device=device)

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Params:
        gen: generator model
        disc: discriminator model
        criterion: Loss function
        real: batch of real images
        num_images: number of images to generate
        z_dim: noise vector dimension
    Returns:
        disc_loss: a torch scalar loss value for the batch
    '''

    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Params:
        gen: generator model
        disc: discriminator model
        criterion: Loss function
        num_images: number of images to generate
        z_dim: noise vector dimension
    Returns:
        disc_loss: a torch scalar loss value for the batch
    '''

    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss

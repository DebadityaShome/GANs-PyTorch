import torch
from torch import nn
torch.manual_seed(0)

def discriminator_block(input_dim, output_dim):
    '''
    Params:
        input_dim: scalar value representing dimension of input vector
        output_dim: scalar value representing dimension of output vector
    Returns:
        a discriminator neural network layer with linear transformation followed by
        a LeakyReLU activation with negative slope of 0.2
    '''

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2),
    )

class Discriminator(nn.Module):
    '''
    Values:
        im_dim: scalar representing dimension of images in the dataset
        hidden_dim: scalar representing inner dimension
    '''
    def __init__(self):
        super(Discriminator).__init__()
        self.disc = nn.Sequential(
            discriminator_block(im_dim, hidden_dim * 4),
            discriminator_block(hidden_dim * 4, hidden_dim * 2),
            discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, image):
        '''
        Params:
            image: a flattened image tensor with dimension = im_dim
        '''
        return self.disc(image)
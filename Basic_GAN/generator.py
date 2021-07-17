import torch
from torch import nn
torch.manual_seed(0)

def generator_block(input_dim, output_dim):
    '''
    Function for returning a block of generator's neural network

    Params:
        input_dim: scalar value representing dimension of input vector
        output_dim: scalar value representing dimension of output vector
    Returns:
        a generator neural network layer with linear transformation followed by BatchNorm and ReLU
    '''

    return nn.Sequential(
        nn.Linear(input_dim, output_dim), 
        nn.BatchNorm1d(output_dim), 
        nn.ReLU(inplace=True),
    )

class Generator(nn.Module):
    '''
        z_dim: scalar representing dimension of the noise vector
        im_dim: scalar representing dimension of the images in the dataset used
        hidden_dim: scalar representing the inner dimension of hidden layer
    '''
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            generator_block(z_dim, hidden_dim), 
            generator_block(hidden_dim, hidden_dim * 2), 
            generator_block(hidden_dim * 2, hidden_dim * 4),
            generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim), 
            nn.Sigmoid(),
        )
    
    def forward(self, noise):
        '''
        Params: noise tensor with dimensions (n_samples, z_dim)
        Returns: generated images
        '''
        return self.gen(noise)
    

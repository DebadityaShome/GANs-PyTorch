from utils import *
from generator import *
from discriminator import *

## Hyperparameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500 
batch_size = 128
lr = 0.00001
device = 'cpu'

# Load MNIST with DataLoader
dataloader = DataLoader(
    MNIST('.', 
    download=True,
    transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# States
cur_step = 0
mean_generator_loss = 0    
mean_discriminator_loss = 0
test_generator = True 
gen_loss = False 
error = False 

for epoch in range(n_epochs):

    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten batch of real images from dataset
        real = real.view(cur_batch_size, -1).to(device)

        # Zero out gradients before backprop
        disc_opt.zero_grad()

        # Discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Update optimizer
        disc_opt.step()

        # Zero out gradients before backprop
        gen_opt.zero_grad()

        # Generator loss
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)

        # Update gradients
        gen_loss.backward()

        # Update optimizer
        gen_opt.step()

        # Keep track of average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        # Monitoring progress
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Epoch {epoch}, step {cur_step} -> Generator loss: {mean_generator_loss}, discriminator_loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1


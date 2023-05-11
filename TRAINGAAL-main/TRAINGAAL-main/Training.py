import torch
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd.variable import Variable
from swat_loader import *
from GANModels import *

# 1. Load and preprocess the SWAT dataset
# Assuming you have a function called `load_swat_data` that returns a PyTorch Dataset
# You might need to customize this function to load the dataset in the format you need


dataset = swat_load_dataset()
batch_size = 64

# 2. Create a DataLoader for training the model
data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 3. Instantiate the Generator and Discriminator
generator = Generator()
discriminator = Discriminator()

# 4. Set up the loss functions and optimizers
lr = 0.0002
g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
d_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)

# 5. Train the GAN, alternating between training the Generator and Discriminator
num_epochs = 100
critic_iterations = 5
clip_value = 0.01

for epoch in range(num_epochs):
    for batch_idx, (real_data, _) in enumerate(data_loader):
        batch_size = real_data.size(0)
        
        # Train the Discriminator
        for _ in range(critic_iterations):
            d_optimizer.zero_grad()

            real_data = Variable(real_data)
            real_preds = discriminator(real_data)
            
            noise = Variable(torch.randn(batch_size, generator.latent_dim))
            fake_data = generator(noise)
            fake_preds = discriminator(fake_data.detach())

            d_loss = -torch.mean(real_preds) + torch.mean(fake_preds)
            d_loss.backward()
            d_optimizer.step()

            # Clip discriminator weights
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)
        
        # Train the Generator
        g_optimizer.zero_grad()
        
        noise = Variable(torch.randn(batch_size, generator.latent_dim))
        fake_data = generator(noise)
        fake_preds = discriminator(fake_data)

        g_loss = -torch.mean(fake_preds)
        g_loss.backward()
        g_optimizer.step()
        
        print("Epoch [{}/{}], Batch [{}/{}], D Loss: {:.4f}, G Loss: {:.4f}".format(
            epoch, num_epochs, batch_idx, len(data_loader), d_loss.item(), g_loss.item()))

# 6. Save the trained model
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

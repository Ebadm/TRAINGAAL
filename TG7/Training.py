import torch
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd.variable import Variable
from swat_loader import *
from GANModels import *
import argparse
import gc
import time

def calculate_uncertainty(scores):
    probabilities = F.softmax(scores, dim=1)
    log_probabilities = F.log_softmax(scores, dim=1)
    uncertainty = -1.0 * (probabilities * log_probabilities).sum(1)
    return uncertainty


def train_gan(enable_cuda=False):
    # Check if CUDA is available and enabled
    if enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        gc.collect()
        torch.cuda.empty_cache()
        print("CUDA is enabled. Training on GPU.")
       
    else:
        device = torch.device('cpu')
        print("Training on CPU.")


    # 1. Load and preprocess the SWAT dataset
    dataset = swat_load_dataset()
    batch_size = 128


    # 2. Create a DataLoader for training the model
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # 3. Instantiate the Generator and Discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)


    # 4. Set up the loss functions and optimizers
    lr = 0.0002
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)


    # 5. Train the GAN, alternating between training the Generator and Discriminator
    num_epochs = 100
    critic_iterations = 5
    clip_value = 0.01
    active_learning_percentage = 0.25  # This means 25% of the most uncertain samples will be used for training.

    for epoch in range(num_epochs):
        for batch_idx, (real_data, _) in enumerate(data_loader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # Train the Discriminator
            for _ in range(critic_iterations):
                d_optimizer.zero_grad()

                real_data = Variable(real_data)
                real_preds = discriminator(real_data)

                # Generate synthetic samples
                noise = Variable(torch.randn(batch_size ,generator.latent_dim)).to(device)
                fake_data = generator(noise)

                #FAKE torch.Size([128, 30, 51])
                #REAL torch.Size([128, 30, 51])

                # Calculate uncertainty of the discriminator
                fake_scores = discriminator(fake_data.detach())
                uncertainties = calculate_uncertainty(fake_scores)

                # Select the most uncertain samples
                _, indices = torch.topk(uncertainties, int(active_learning_percentage * batch_size))
                selected_fake_data = fake_data[indices]

                # Compute loss using the selected samples
                selected_fake_preds = discriminator(selected_fake_data)
                d_loss = -torch.mean(real_preds) + torch.mean(selected_fake_preds)
                d_loss.backward()
                d_optimizer.step()

                # Clip discriminator weights
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Train the Generator
            g_optimizer.zero_grad()

            noise = Variable(torch.randn(batch_size, generator.latent_dim)).to(device)
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


# You can call your function with the desired argument here
if __name__ == "__main__":
    # Define and parse command line arguments
    parser = argparse.ArgumentParser(description='GAN training script')
    parser.add_argument('--enable_cuda', type=bool, default=False,
                        help='Enable CUDA (default: False)')
    args = parser.parse_args()

    start_time = time.time()
    # Enable or disable CUDA based on command line argument
    train_gan(enable_cuda=args.enable_cuda)

    end_time = time.time()
    print("Elapsed Time: {:.2f} seconds".format(end_time - start_time))



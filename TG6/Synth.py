from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.functional import kl_div
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch
from GANModels import *
from swat_loader import *
from torch.autograd.variable import Variable
import numpy as np
import os
import numpy as np
from scipy.special import kl_div
import matplotlib.pyplot as plt
from visualizationMetrics import visualization
from torch.nn.functional import kl_div
from sklearn.metrics import f1_score, precision_score, recall_score

class Synthetic_Dataset(Dataset):
    def __init__(self, generator_path='generator.pth', device=torch.device('cpu'), threshold=0.1):
        self.gen_net = Generator().to(device)
        self.device = device
        self.threshold = threshold
        generator = torch.load(generator_path, map_location=self.device)
        self.gen_net.load_state_dict(generator)
        self.gen_net.eval()  # Set the generator in evaluation mode

        # Load the test data
        self.dataset = swat_load_dataset(is_train=False)

        # DataLoader
        self.data_loader = DataLoader(self.dataset, batch_size=128, shuffle=False, num_workers=0)

        # To store KL divergence values for each batch
        self.kl_divergences = []

    def calculate_metrics(self):
        for batch_idx, (real_data, _) in enumerate(self.data_loader):
            real_data = real_data.to(self.device)
            batch_size, sliding_window_length, features_size = real_data.size()

            # Compute the average across the sliding window dimension
            real_data_avg = torch.mean(real_data, dim=1)

            # Reshape the tensor to have shape (batch_size, features_size)
            real_data_avg = real_data_avg.view(batch_size, features_size)

            # Generate synthetic sample
            synthetic_sample = self.gen_net(real_data_avg)

            # Calculate KL divergence and store it
            kl_divergence = kl_div(real_data, synthetic_sample).sum()
            self.kl_divergences.append(kl_divergence.item())

            # Detach the tensors and convert them to numpy for visualization
            real_data = real_data.cpu().detach().numpy()
            synthetic_sample = synthetic_sample.cpu().detach().numpy()

            # Visualize the real and synthetic data
            #visualization(real_data, synthetic_sample, 'tsne', 'Running-tsne')

        # Calculate and print the average KL divergence
        avg_kl_divergence = np.mean(self.kl_divergences)
        print("Average KL Divergence across all batches: ", avg_kl_divergence)

device = torch.device('cpu')
print("Training on CPU.")
synthetic_dataset = Synthetic_Dataset(device=device)
synthetic_dataset.calculate_metrics()


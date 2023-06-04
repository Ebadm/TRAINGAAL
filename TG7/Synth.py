from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch
from GANModels import *
from swat_loader import *
from torch.autograd.variable import Variable
import numpy as np
import os
from visualizationMetrics import visualization
import numpy as np
from scipy.special import kl_div
import matplotlib.pyplot as plt


def calculate_mse(array1, array2):
    assert array1.shape == array2.shape, "Arrays must have the same shape"
    squared_diff = np.square(array1 - array2)
    mse = np.mean(squared_diff)
    return mse


class Synthetic_Dataset(Dataset):
    def __init__(self):
        gen_net = Generator().to(device)
        generator = torch.load('generator.pth', map_location=device)
        gen_net.load_state_dict(generator)
        dataset = swat_load_dataset(is_train=True, is_attack=True)
        data_loader = data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        anomaly_scores = []
        for batch_idx, (real_data, _) in enumerate(data_loader):
            real_data = real_data.to(device)
            real_data = Variable(real_data).to(device)
            batch_size, sliding_window_length, features_size = real_data.size()
            real_data_avg = torch.mean(real_data, dim=1)
            real_data_avg = real_data_avg.view(batch_size, features_size)
            synthetic_sample = gen_net(real_data_avg)

            real_data = real_data.cpu().detach().numpy()
            synthetic_sample = synthetic_sample.cpu().detach().numpy()

            # Calculate reconstruction error (MSE)
            reconstruction_error = np.mean(np.square(real_data - synthetic_sample), axis=(1, 2))
            print(np.mean(reconstruction_error))
            # Append anomaly scores to the list
            anomaly_scores.append(np.mean(reconstruction_error))
            print("Batch [{}/{}], Mean Squared Error: {:.4f}".format(batch_idx, len(data_loader), np.mean(reconstruction_error)))
            #visualization(real_data, synthetic_sample, 'tsne', 'Running-tsne')
        print("AVERAGE: ",sum(anomaly_scores) / len(anomaly_scores))

enable_cuda = True
if enable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is enabled. Training on GPU.")
else:
    device = torch.device('cpu')
    print("Training on CPU.")

synthentic_dataset = Synthetic_Dataset()



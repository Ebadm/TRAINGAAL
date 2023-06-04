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
import gc


def calculate_mse(array1, array2):
    assert array1.shape == array2.shape, "Arrays must have the same shape"
    squared_diff = np.square(array1 - array2)
    mse = np.mean(squared_diff)
    return mse


class Synthetic_Dataset(Dataset):
    def __init__(self):
        gen_net = Generator(seq_len=60).to(device)
        generator = torch.load('generator.pth', map_location=device)
        gen_net.load_state_dict(generator)
        dis_net = Discriminator(seq_len=60).to(device)
        discriminator = torch.load('discriminator.pth', map_location=device)
        gen_net.load_state_dict(generator)
        dis_net.load_state_dict(discriminator)
        dataset = swat_load_dataset(is_train=True, is_attack=True)
        data_loader = data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        anomaly_scores = []
        with torch.no_grad():
            for batch_idx, (real_data, _) in enumerate(data_loader):
                real_data = real_data.to(device)
                real_data = Variable(real_data).to(device)
                batch_size, sliding_window_length, features_size = real_data.size()
                real_data_avg = torch.mean(real_data, dim=1)
                real_data_avg = real_data_avg.view(batch_size, features_size)
                synthetic_sample = gen_net(real_data_avg)
                discriminator_output_real = dis_net(real_data)
                discriminator_output_syn = dis_net(synthetic_sample)
                d_loss = -torch.mean(discriminator_output_real) + torch.mean(discriminator_output_syn)
                g_loss = -torch.mean(discriminator_output_syn)
                alpha = 0.3
                ads_loss = (1 - alpha) * d_loss + alpha * g_loss
                anomaly_scores.append(ads_loss.item())

                real_data = real_data.cpu().detach().numpy()
                synthetic_sample = synthetic_sample.cpu().detach().numpy()
                print("Batch [{}/{}], ADS: {:.4f}".format(batch_idx, len(data_loader), ads_loss.item()))
                # visualization(real_data, synthetic_sample, 'tsne', 'Running-tsne')

        anomaly_scores = np.array(anomaly_scores)
        print("AVERAGE: ", np.mean(anomaly_scores))
        print("MAX: ", np.max(anomaly_scores))
        print("MIN: ", np.min(anomaly_scores))
        

enable_cuda = True
# Check if CUDA is available and enabled
if enable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
    gc.collect()
    torch.cuda.empty_cache()
    print("CUDA is enabled. Training on GPU.")
    
else:
    device = torch.device('cpu')
    print("Training on CPU.")

synthetic_dataset = Synthetic_Dataset()

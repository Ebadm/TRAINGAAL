from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from visualizationMetrics import visualization
import torch
from GANModels import *
from swat_loader import *
from torch.autograd.variable import Variable
import numpy as np
import os
import numpy as np
from scipy.special import kl_div
import matplotlib.pyplot as plt
from scipy.special import kl_div

class Synthetic_Dataset(Dataset):
    def __init__(self):
        gen_net=Generator()
        generator = torch.load('generator.pth', map_location=torch.device('cpu'))
        gen_net.load_state_dict(generator)
        dataset=swat_load_dataset(is_train=False)
       
        data_loader = data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        for batch_idx, (real_data,_) in enumerate(data_loader):
            real_data = real_data.to(device)
            real_data=Variable(real_data).to(device)
            batch_size, sliding_window_length, features_size = real_data.size()
            # Compute the average across the sliding window dimension
            real_data_avg = torch.mean(real_data, dim=1)
            # Reshape the tensor to have shape (batch_size, features_size)
            real_data_avg = real_data_avg.view(batch_size, features_size)
            synthetic_sample = gen_net(real_data_avg)
            real_data = real_data.cpu().detach().numpy()
            synthetic_sample = synthetic_sample.cpu().detach().numpy()
            visualization(real_data, synthetic_sample, 'tsne', 'Running-tsne')
            break


device = torch.device('cpu')
print("Training on CPU.")

synthentic_dataset = Synthetic_Dataset()        

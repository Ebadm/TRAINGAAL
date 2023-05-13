from torch.utils.data import Dataset, DataLoader
import torch
from GANModels import Generator
from swat_loader2 import swat_load_dataset
import numpy as np
from scipy.special import kl_div
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

class Synthetic_Dataset(Dataset):
    def __init__(self, sample_size=1000):
        self.sample_size = sample_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator().to(self.device)
        try:
            self.generator.load_state_dict(torch.load('generator.pth'))
        except Exception as e:
            print("Error loading the model: ", e)
        self.generator.eval() 
        self.real_data = swat_load_dataset(is_train=False)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        test_data, _ = self.real_data[idx]  # Unpack to ignore the dummy target
        test_data = test_data.to(device)  # Move test_data to the same device as the generator
        print(test_data.shape)
        batch_size = 1 
        sliding_window_length, features_size = test_data.size()
        test_data_avg = torch.mean(test_data, dim=1)
        test_data_avg = test_data_avg.view(batch_size, features_size)
        print(test_data_avg.shape)
        with torch.no_grad():
            synthetic_sample = self.generator(test_data_avg)
        return test_data, synthetic_sample


def calculate_metrics(real_data, synthetic_data, threshold):
    kl_divergence = kl_div(real_data, synthetic_data).sum(axis=1)
    anomalies = kl_divergence > threshold
    f1 = f1_score(anomalies, np.ones_like(anomalies))
    precision = precision_score(anomalies, np.ones_like(anomalies))
    recall = recall_score(anomalies, np.ones_like(anomalies))
    return kl_divergence, anomalies, f1, precision, recall

def plot_data(data, title, y_label):
    plt.scatter(range(len(data)), data, c=data, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('Data Point')
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on {}.".format(device))

synthetic_dataset = Synthetic_Dataset()
synthetic_dataset.generator = synthetic_dataset.generator.to(device)
data_loader = DataLoader(synthetic_dataset, batch_size=128, shuffle=False, num_workers=0)

threshold = 0.1  # Set your desired threshold value here

for batch_idx, (real_data, synthetic_data) in enumerate(data_loader):
    real_data = real_data.to(device)
    synthetic_data = synthetic_data.to(device)
    real_data_np = real_data.cpu().numpy()
    synthetic_data_np = synthetic_data.cpu().numpy()

    kl_divergence, anomalies, f1, precision, recall = calculate_metrics(real_data_np, synthetic_data_np, threshold)

    print("Batch [{}/{}]: KL Divergence: {}, F1 Score: {}, Precision: {}, Recall: {}".format(
        batch_idx, len(data_loader), kl_divergence, f1, precision, recall))

    plot_data(kl_divergence, 'KL Divergence between Arrays', 'KL Divergence')
    plot_data(anomalies, 'Anomaly Detection', 'Anomaly')

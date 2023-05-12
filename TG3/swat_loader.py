import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class swat_load_dataset(Dataset):
    def __init__(self, verbose=True, is_normalize=False, one_hot_encode=True, data_mode='Train'):
        super().__init__()
        self.verbose = verbose
        # Download and unzip original dataset
        if not os.path.isfile('./swat-2015-data.zip'):
            print("Downloading swat-2015-data.zip file")
            # redirect link https://www.dropbox.com/s/raw/qoy6q9uj52h9gt6/swat-2015-data.zip
            self.download_url('https://www.dropbox.com/s/raw/qoy6q9uj52h9gt6/swat-2015-data.zip', './swat-2015-data.zip')
        if not os.path.exists('./swat-2015-data.npy'):
            # print("Current working directory:", os.getcwd())
            os.listdir('.')
            shutil.unpack_archive('./swat-2015-data.zip', '.', 'zip')
            print("Successfully Unpacked")


        window_size = 30  # Set this to the desired window size


        # df = pd.read_excel(xlsx_file, engine='openpyxl', header=1)  # Read the file
        x_train = np.load('swat-2015-data.npy')
        x_train = x_train[:7000, :]


        # Create sliding windows
        x_train_windows = self.create_windows(x_train, window_size)


        # Convert to PyTorch tensors
        self.x_train = torch.tensor(x_train_windows, dtype=torch.float32)


        if self.verbose:
            print("Shape of Data", self.x_train.shape)


    def create_windows(self, data, window_size):
        """
        Creates a sliding window view of the data.
        """
        num_samples, num_features = data.shape
        windowed_data = []


        # Slide the window over the number of samples (data points)
        for start in range(num_samples - window_size + 1):
            end = start + window_size
            window = data[start:end, :]
            windowed_data.append(window)


        windowed_data = np.stack(windowed_data)


        return windowed_data


    def __getitem__(self, index):
        # Get a window of data at the given index
        window = self.x_train[index]


        # Use a dummy target since it's not being used in your training loop
        dummy_target = torch.tensor(0)


        return window, dummy_target


    def __len__(self):
        return len(self.x_train)


    def download_url(self, url, destination):
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

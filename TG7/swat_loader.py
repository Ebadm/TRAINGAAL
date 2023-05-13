import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class swat_load_dataset(Dataset):
    def __init__(self, is_train=True, split_ratio=0.9, verbose=True, is_attack=False):
        super().__init__()
        self.verbose = verbose
        self.is_train = is_train
        self.window_size = 60  # Set this to the desired window size
        if is_attack:
            # Download and unzip original dataset
            if not os.path.isfile('./swat_attack_data.zip'):
                print("Downloading swat_attack_data.zip file")
                # redirect link https://www.dropbox.com/s/raw/w0lzvo74wgfhuox/swat_attack_data.zip
                self.download_url('https://www.dropbox.com/s/raw/w0lzvo74wgfhuox/swat_attack_data.zip', './swat_attack_data.zip')
            if not os.path.exists('./attack_data.npy'):
                os.listdir('.')
                shutil.unpack_archive('./swat_attack_data.zip', '.', 'zip')
                print("Successfully Unpacked")
            x_train = np.load('attack_data.npy', allow_pickle=True)
            print(x_train.shape)
            exit()

        else:
            # Download and unzip original dataset
            if not os.path.isfile('./swat-2015-data.zip'):
                print("Downloading swat-2015-data.zip file")
                # redirect link https://www.dropbox.com/s/raw/qoy6q9uj52h9gt6/swat-2015-data.zip
                self.download_url('https://www.dropbox.com/s/raw/qoy6q9uj52h9gt6/swat-2015-data.zip', './swat-2015-data.zip')
            if not os.path.exists('./swat-2015-data.npy'):
                os.listdir('.')
                shutil.unpack_archive('./swat-2015-data.zip', '.', 'zip')
                print("Successfully Unpacked")
            x_train = np.load('swat-2015-data.npy')
            x_train = x_train[:10000, :]


        # Create sliding windows
        x_train_windows = self.create_windows(x_train, self.window_size)

        # Split the data into training and test sets
        split_idx = int(split_ratio * len(x_train_windows))

        if is_train:
            self.data = x_train_windows[:split_idx]
        else:
            self.data = x_train_windows[split_idx:]

        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)


        if self.verbose:
            print("Shape of Data", self.data.shape)


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
        window = self.data[index]
        # Use a dummy target since it's not being used in your training loop
        dummy_target = torch.tensor(0)
        return window, dummy_target


    def __len__(self):
        return len(self.data)


    def download_url(self, url, destination):
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


    def pre_process_attack(self, data):
        last_column = data[:, -1]  # Extract the last column
        last_column[last_column == 'Normal'] = 0
        last_column[last_column == 'Attack'] = 1
        # Update the modified column back into the original array
        data[:, -1] = last_column
        data_normal = data[last_column == 0]
        data_attack = data[last_column == 1]
        labels_normal = data_normal[:,-1]
        labels_attack = data_attack[:,-1]
        data_normal = data_normal[:, :-1]
        data_attack = data_attack[:, :-1]
        data_normal_windows = self.create_windows(data_normal, self.window_size)
        data_attack_windows = self.create_windows(data_attack, self.window_size)

        concatenated_data = np.concatenate((data_normal, data_attack), axis=0)

        return data_normal, data_attack, labels_normal, labels_attack



swat_load_dataset(is_attack=True)
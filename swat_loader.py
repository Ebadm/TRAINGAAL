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
        self.one_hot_encode = one_hot_encode
        self.data_mode = data_mode
        self.is_normalize = is_normalize

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

        # Load the xlsx file
        # xlsx_file = 'SWaT_Dataset_Normal_v0.xlsx'
        # df = pd.read_excel(xlsx_file, engine='openpyxl', header=1)  # Read the file
        x_train = np.load('swat-2015-data.npy')
        df = pd.DataFrame(x_train)

        if self.verbose:
            print("Shape of Data", df.shape)

        # reshape x_train, x_test data shape from (BH, length, channel) to (BH, channel, 1, length)
        # x_train has shape (BH, width)
        BH, width = x_train.shape
        # reshape x_train
        self.x_train = torch.tensor(x_train.reshape(BH, 1, 1, width), dtype=torch.float32)
        #self.x_train = torch.tensor(x_train, dtype=torch.float32)

        
        print(f'x_train shape is {self.x_train.shape}')

    def __getitem__(self, index):
        # Get data sample at the given index
        data_sample = self.x_train[index]
        
        # Use a dummy target since it's not being used in your training loop
        dummy_target = torch.tensor(0)

        return data_sample, dummy_target

    def __len__(self):
        return len(self.x_train)

    def download_url(self, url, destination):
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

import torch
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split


class DataModule(LightningDataModule):
    def __init__(
        self,
        DATA_PATH=None,
        LABEL_PATH=None,
        batch_size=1,
        train_size=1/7,
        num_workers=1,
        multiclass=False,
    ):
        super().__init__()
        self.DATA_PATH = DATA_PATH
        self.LABEL_PATH = LABEL_PATH
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers
        self.multiclass = multiclass

    def prepare_data(self):
        
        def preprocess(df):
            stress = np.array([df[col] for col in df.columns if "stress" in col])
            strain = np.array([df[col] for col in df.columns if "strain" in col])
            exp_strain = strain[0]
            x_min, x_max = 0.002, exp_strain.max()
            prune = np.logical_and(strain[1] > x_min, strain[1] < x_max)  # Indices to prune simulated curves by
            sim_strain = strain[1:,:][:,prune]
            sim_stress = stress[1:,:][:,prune]
            return sim_strain, sim_stress
        
        self.df = pd.read_csv(self.DATA_PATH)
        sim_strain, sim_stress = preprocess(self.df)
        self.strain = sim_strain[0]
        self.df = np.array(sim_stress)
        
        self.labels = np.array([[200,1000,1000,1],
                                [600,500,10000,1],
                                [1000,100,5000,0.5],
                                [200,200,1000,2],
                                [300,200,2000,3],
                                [500,1000,4000,2],
                                [300,2000,3000,3]])
        print(self.df.shape)
        print(self.labels.shape)


    def setup(self, stage=None):
        self.train = TensorDataset(torch.Tensor(self.labels[:4]), torch.Tensor(self.df[:4]),)
        self.val = TensorDataset(torch.Tensor(self.labels[5:]), torch.Tensor(self.df[5:]))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)
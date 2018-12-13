import pandas as pd
import numpy as np


class DataLoader:
    """
    Class I used for data loading. Does not really help me much anymore.
    """
    def __init__(self):
        self.images = None
        self.ids_ = None
        self.target = None
        self.patients = None
        self.loaded = False

    def load_data(self, path):
        print("Loading data")
        self.images = np.load(f"{path}/images.npz")['arr_0']
        self.ids_ = np.load(f"{path}/ids.npy")
        self.target = np.load(f"{path}/target.npy")
        self.patients = pd.read_csv(f"{path}/patients.csv")
        print("Done.")
        self.loaded = True
        return self.images, self.ids_, self.target, self.patients

    def is_loaded(self):
        if self.loaded:
            return self.loaded
        else:
            return self.loaded
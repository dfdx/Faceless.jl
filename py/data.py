
import h5py
import numpy as np
from os.path import expanduser

def load_data(filename=expanduser("~/data/CK/dataset_10708.h5")):
    with h5py.File(filename,'r') as hf:
        data = hf.get('dataset')
        np_data = np.array(data)
        return np_data


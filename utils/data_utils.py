import os
import pickle
import numpy as np


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    # if extension is txt, read as text and convert to numpy array
    if os.path.splitext(filename)[1] == ".npy":
        # eval it as numpy array
        return np.load(filename, allow_pickle=True)
    
    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def load_heatmaps(filename, symmetric=True):
    if filename is None:
        return None
    heatmaps, *_ = load_dataset(filename)
    if (heatmaps >= 0).all():
        print("Warning: heatmaps where not stored in logaritmic space, conversion may be lossy!")
        heatmaps = np.log(heatmaps)
    return heatmaps if not symmetric else np.maximum(heatmaps, np.transpose(heatmaps, (0, 2, 1)))

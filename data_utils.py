import numpy as np
import scipy.sparse as sp


def loadRedditFromNPZ(dataset_dir, add_self_loop=True):
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")
    adj = adj + adj.T
    if add_self_loop:
        adj += sp.eye(adj.shape[0])
    return adj.asformat("lil"), data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], \
           data['test_index']

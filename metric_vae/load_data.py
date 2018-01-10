import numpy as np
import os
from six.moves import cPickle
import scipy.io

class DataLoader(object):
    def __init__(self, data, batch_size, n_batches=None):
        self.data = data
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self):
        self.cur_ind = 0
        self.inds = np.arange(self.data['labels'].shape[0]).astype(np.int32)
        np.random.shuffle(self.inds)
        return self

    def __next__(self):
        if (self.n_batches is not None and self.cur_ind // self.batch_size >= self.n_batches) or self.cur_ind > self.data['labels'].shape[0]:
            raise StopIteration
        else:
            images = self.data['images'][self.inds[self.cur_ind:self.cur_ind+self.batch_size]]
            if self.data['proxy'] is not None:
                proxy = self.data['proxy'][self.inds[self.cur_ind:self.cur_ind+self.batch_size]]
            else:
                proxy = [None]
            self.cur_ind += self.batch_size
            return images, proxy

    def next(self):
        return self.__next__()

def load_tfd():
    data_dir = 'data/'

    def preprocess(img_array):
        return np.float32(img_array) / 255.0

    # d = np.load(os.path.join(data_dir, 'tfd_data_array.npz'))
    # train = preprocess(d['images'][d['train_inds']])
    # val = preprocess(d['images'][d['val_inds']])
    #
    # return train, val

    d = np.load(os.path.join(data_dir, 'tfd_data_array.npz'))
    proxy = np.load(os.path.join(data_dir, 'tfd_data_proxy_distance.npz'))
    fau_ds = np.load(os.path.join(data_dir, 'fau_levels.npz'))
    example_inds = fau_ds['inds']

    exprs = scipy.io.loadmat(os.path.join(data_dir, 'TFD_ranzato_96x96.mat'))['labs_ex'][example_inds].reshape(-1)
    images = d['images'][example_inds]
    proxy = proxy['data'][example_inds]


    filt_inds = np.where(np.greater(exprs, 0))
    exprs = exprs[filt_inds]
    images = images[filt_inds]
    proxy = proxy[filt_inds]

    n_examples = exprs.shape[0]

    exprs += -1
    s = np.zeros((n_examples, 7))
    for i in range(n_examples):
        s[i, exprs[i]] = 1
    exprs_onehot = s

    TRAIN_FRAC = 0.8
    n_train = int(n_examples * TRAIN_FRAC)
    n_val = int(0.5 * (n_examples - n_train))
    inds = np.arange(n_examples)
    train_ind = inds[:n_train]
    val_ind = inds[n_train: n_train + n_val]

    train_im = preprocess(images[train_ind])
    val_im = preprocess(images[val_ind])

    train_proxy = proxy[train_ind]
    val_proxy = proxy[val_ind]

    train_exprs = exprs[train_ind]
    val_exprs = exprs[val_ind]

    train_exprs_onehot = exprs_onehot[train_ind]
    val_exprs_onehot = exprs_onehot[val_ind]

    train = {'images': train_im, 'labels_onehot': train_exprs_onehot, 'labels': train_exprs, 'proxy': train_proxy}
    val = {'images': val_im, 'labels_onehot': val_exprs_onehot, 'labels': val_exprs, 'proxy': val_proxy}

    return train, val



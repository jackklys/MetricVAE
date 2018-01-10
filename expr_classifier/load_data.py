import scipy.io
import numpy as np
import os
from six.moves import cPickle

class DataLoader(object):
    def __init__(self, data, batch_size, n_batches=None):
        self.data = data
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self):
        self.cur_ind = 0
        self.inds = np.arange(self.data['images'].shape[0]).astype(np.int32)
        np.random.shuffle(self.inds)
        return self

    def __next__(self):
        if (self.n_batches is not None and self.cur_ind // self.batch_size >= self.n_batches) or self.cur_ind > self.data['images'].shape[0]:
            raise StopIteration
        else:
            ret_im = self.data['images'][self.inds[self.cur_ind:self.cur_ind+self.batch_size]]
            ret_label = self.data['labels_onehot'][self.inds[self.cur_ind:self.cur_ind + self.batch_size]]
            self.cur_ind += self.batch_size
            return ret_im, ret_label

    def next(self):
        return self.__next__()

class DataLoader2(object):
    def __init__(self, data, batch_size, n_batches=None):
        self.data = data
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self):
        self.cur_ind = 0
        self.inds = np.arange(self.data.shape[0]).astype(np.int32)
        return self

    def __next__(self):
        if (self.n_batches is not None and self.cur_ind // self.batch_size >= self.n_batches) or self.cur_ind > self.data.shape[0]:
            raise StopIteration
        else:
            ret_im = self.data[self.inds[self.cur_ind:self.cur_ind+self.batch_size]]
            self.cur_ind += self.batch_size
            return ret_im

    def next(self):
        return self.__next__()

def load_tfd():
    data_dir = 'data/'

    def preprocess(img_array):
        return np.float32(img_array) / 255.0

    d = np.load(os.path.join(data_dir, 'tfd_data_array.npz'))
    fau_ds = np.load(os.path.join(data_dir, 'fau_levels.npz'))
    example_inds = fau_ds['inds']

    exprs = scipy.io.loadmat(os.path.join(data_dir, 'TFD_ranzato_96x96.mat'))['labs_ex'][example_inds].reshape(-1)
    images = d['images'][example_inds]

    filt_inds = np.where(np.greater(exprs, 0))
    exprs = exprs[filt_inds]
    images = images[filt_inds]

    n_examples = exprs.shape[0]

    exprs += -1
    s = np.zeros((n_examples, 7))
    for i in range(n_examples):
        s[i, exprs[i]] = 1
    exprs_onehot = s

    TRAIN_FRAC = 0.8
    n_train = int(n_examples * TRAIN_FRAC)
    n_test = int(0.5 * (n_examples - n_train))
    inds = np.arange(n_examples)
    train_ind = inds[:n_train]
    test_ind = inds[n_train : n_train + n_test]

    train_im = preprocess(images[train_ind])
    test_im = preprocess(images[test_ind])

    train_exprs = exprs[train_ind]
    test_exprs = exprs[test_ind]

    train_exprs_onehot = exprs_onehot[train_ind]
    test_exprs_onehot = exprs_onehot[test_ind]

    train = {'images': train_im, 'labels_onehot': train_exprs_onehot, 'labels': train_exprs}
    test = {'images': test_im, 'labels_onehot': test_exprs_onehot, 'labels': test_exprs}

    return train, test

from nn import Classifier
from train import run_train
from load_data import load_tfd, DataLoader, DataLoader2
import scipy
import tsne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import os
import tensorflow as tf
import keras
import keras.backend as K
from visualize import visualize_grid

def apply_proxy(sess, model):
    def preprocess(img_array):
        return np.float32(img_array) / 255.0

    data_dir = 'data/'
    data = np.load(os.path.join(data_dir, 'tfd_data_array.npz'))
    data = preprocess(data['images'])
    data_loader2 = DataLoader2(data, 1000)

    images = []
    for i, batch in enumerate(data_loader2):
        h, z, softz = sess.run(model.forward(batch, penultimate=True), {K.learning_phase(): 0})
        images.append(z)
    images = np.concatenate(images, axis=0)
    np.savez('data/tfd_data_proxy_distance.npz', data=images)

def order_expr(sess, model, data, save_dir):
    x = data['images']
    y = data['labels']
    z = sess.run(model.forward(x), {K.learning_phase(): 0})
    u = []
    e = []
    for i in range(7):
        ind = np.where(y == i)[0]
        expr = x[ind]
        expr_z = z[ind]
        intensity = expr_z[:,i][:14]
        e.append(expr[np.argsort(intensity)])
        u.append([np.sort(intensity).reshape(1,-1)])
    # np.savetxt('intensities.txt', np.asarray(np.bmat(u)))
    u = np.asarray(np.bmat(u))
    plt.plot(u, 'o')
    plt.savefig(save_dir + 'intensities.png')
    plt.close()
    scipy.misc.imsave(save_dir + 'ordered_samples.png', visualize_grid(np.concatenate(e, axis=0), num_rows=7))

def train(save_dir):
    np.random.seed(1234)
    batch_size = 64
    n_epochs = 20
    n_batches = 100
    ks = 5
    z_dim = 7
    h_dim = 1024
    rate = 0.3
    learning_rate = [0.001, 0.0001]

    train_data, val_data = load_tfd()
    image_size = train_data['images'].shape[-1]

    train_loader = DataLoader(train_data, batch_size)
    val_loader = DataLoader(val_data, batch_size, n_batches)

    layers = [{'type': tf.reshape, 'shape': [-1, image_size, image_size, 1]},

                keras.layers.Conv2D(filters=64, kernel_size=ks, strides=1, padding='same'),
                keras.layers.pooling.MaxPooling2D(),
                keras.layers.normalization.BatchNormalization(),
                {'type': tf.nn.relu},

                keras.layers.Conv2D(filters=128, kernel_size=ks, strides=1, padding='same'),
                keras.layers.pooling.MaxPooling2D(),
                keras.layers.normalization.BatchNormalization(),
                {'type': tf.nn.relu},

                keras.layers.Conv2D(filters=192, kernel_size=ks, strides=1, padding='same'),
                keras.layers.normalization.BatchNormalization(),
                {'type': tf.nn.relu},

                keras.layers.Conv2D(filters=192, kernel_size=ks, strides=1, padding='same'),
                keras.layers.normalization.BatchNormalization(),
                {'type': tf.nn.relu},

                keras.layers.Conv2D(filters=64, kernel_size=ks, strides=1, padding='same'),
                keras.layers.pooling.MaxPooling2D(),
                keras.layers.normalization.BatchNormalization(),
                {'type': tf.nn.relu},

                {'type': tf.contrib.layers.flatten},

                keras.layers.Dense(h_dim),
                keras.layers.normalization.BatchNormalization(),
                {'type': tf.nn.relu},

                keras.layers.Dense(h_dim),
                keras.layers.normalization.BatchNormalization(),
                {'type': tf.nn.relu},

                keras.layers.Dense(z_dim),
                # {'type': tf.reshape, 'shape': [-1, z_dim]}
              ]

    x = tf.placeholder(tf.float32, [None, image_size, image_size], name='x')
    y = tf.placeholder(tf.float32, [None, z_dim], name='y')
    lr = tf.placeholder(tf.float32, name='lr')
    model = Classifier(layers)
    loss = model.loss(y, x)
    correct_prediction = tf.equal(tf.argmax(model.forward(x), 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def collect_updates(layers):
        update_ops = []
        for layer in layers:
            if isinstance(layer, keras.engine.Layer):
                update_ops += layer.updates
        return update_ops

    update_ops = collect_updates(layers)

    step = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for ie in range(n_epochs):
            print("Epoch {:d}/{:d}".format(ie + 1, n_epochs))
            alpha = learning_rate[0] * (1 - float(ie)/n_epochs) + learning_rate[1] * float(ie)/n_epochs

            for ib, data in enumerate(train_loader):
                sample = data[0]
                label = data[1]
                _, _, output = sess.run([step, update_ops, loss],
                                             {x: sample, y:label, lr: alpha,  K.learning_phase(): 1})

            acc_test = sess.run(accuracy, {x: val_data['images'], y: val_data['labels_onehot'], K.learning_phase(): 0})
            acc_train = sess.run(accuracy, {x: train_data['images'], y: train_data['labels_onehot'], K.learning_phase(): 0})
            print('train accuracy: ' + str(acc_train))
            print('test accuracy: ' + str(acc_test))

        h, z, softz = sess.run(model.forward(val_data['images'], penultimate=True), {K.learning_phase(): 0})

        Y = tsne.tsne(h, 2, 50, 20.0)
        plt.scatter(Y[:, 0], Y[:, 1], c=val_data['labels'])
        plt.colorbar(ticks=np.arange(1, 7))
        plt.savefig(save_dir + 'secondlast.png')
        plt.close()

        Y = tsne.tsne(z, 2, 50, 20.0)
        plt.scatter(Y[:, 0], Y[:, 1], c=val_data['labels'])
        plt.colorbar(ticks=np.arange(1, 7))
        plt.savefig(save_dir + 'last.png')
        plt.close()

        Y = tsne.tsne(softz, 2, 50, 20.0)
        plt.scatter(Y[:, 0], Y[:, 1], c=val_data['labels'])
        plt.colorbar(ticks=np.arange(1, 7))
        plt.savefig(save_dir + 'softlast.png')
        plt.close()

        order_expr(sess, model, val_data, save_dir)
        apply_proxy(sess, model)




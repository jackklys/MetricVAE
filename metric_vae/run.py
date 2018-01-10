import tensorflow as tf
import numpy as np
import os
from scipy import misc
from nn.model import VAE
from load_data import load_tfd, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tsne

import keras
import keras.backend as K

def latent_expr(sess, model, data, save_dir):
    z = sess.run(model.encode(data['images']).mu, {model.alpha: 0, model.beta: 0, K.learning_phase(): 0})
    Y = tsne.tsne(z, 2, 50, 20.0)
    plt.scatter(Y[:, 0], Y[:, 1], c = data['labels'])
    plt.colorbar(ticks=np.arange(1, 7))
    plt.savefig(save_dir + 'latent_expr.png')
    plt.close()

def format_log_string(output, fields):
    ret = []
    for field in fields:
        ret.append("{:s} = {:0.6f}".format(field, output[field]))
    return ', '.join(ret)

def merge_dict(x, y):
    ret = x.copy()

    for k,v in y.items():
        ret[k] = v

    return ret

def evaluate(sess, x, y, loss, data_loader, fields, feed):
    ret = {k: 0.0 for k in fields}
    n_batches = 0

    for sample in data_loader:
        images = sample[0]
        proxy_labels = sample[1]
        n_batches += 1
        _, output = sess.run(loss, merge_dict({ x: images, y: proxy_labels }, feed))
        for field in fields:
            ret[field] += output[field]

    for field in fields:
        ret[field] /= n_batches

    return ret

def train(save_dir):
    np.random.seed(1234)
    batch_size = 64
    n_epochs = 50
    ks = 5
    z_dim = 512
    h_dim = 256
    alphas = [0.1, 0.1]
    betas = [1e5, 1e5]
    log_fields = ['loss', 'recon', 'kl', 'dr']
    '''
    possible regularizers:
    none
    pointwise
    expanding_pointwise
    contracting_pointwise
    batchwise
    '''
    distance_regularizer = 'batchwise'
    parameters = [distance_regularizer, n_epochs, ks, z_dim, h_dim, alphas, betas]

    train_data, val_data = load_tfd()

    image_size = train_data['images'].shape[-1]
    down_image_size = image_size/8
    if train_data['proxy'] is not None:
        proxy_size = [None, train_data['proxy'].shape[1]]
    else:
        proxy_size = [None]
        distance_regularizer = 'none'

    train_loader = DataLoader(train_data, batch_size)
    val_loader = DataLoader(val_data, batch_size)

    layers_encode = [ {'type': tf.reshape, 'shape': [-1, image_size, image_size, 1]},

                      keras.layers.Conv2D(filters=64, kernel_size=ks, strides=1, padding='same'),
                      keras.layers.pooling.MaxPooling2D(),
                      keras.layers.normalization.BatchNormalization(),
                      {'type': tf.nn.relu},

                      keras.layers.Conv2D(filters=128, kernel_size=ks, strides=1, padding='same'),
                      keras.layers.pooling.MaxPooling2D(),
                      keras.layers.normalization.BatchNormalization(),
                      {'type': tf.nn.relu},

                      keras.layers.Conv2D(filters=h_dim, kernel_size=ks, strides=1, padding='same'),
                      keras.layers.pooling.MaxPooling2D(),
                      keras.layers.normalization.BatchNormalization(),
                      {'type': tf.nn.relu},

                      {'type': tf.contrib.layers.flatten},
                      keras.layers.Dense(z_dim * 2),
                      {'type': tf.reshape, 'shape': [-1, z_dim, 2]}
                    ]

    layers_decode = [ keras.layers.Dense(down_image_size * down_image_size * h_dim),
                      keras.layers.normalization.BatchNormalization(),
                      {'type': tf.nn.relu},
                      {'type': tf.reshape, 'shape': [-1, down_image_size, down_image_size, h_dim]},

                      keras.layers.Conv2DTranspose(filters=h_dim, kernel_size=ks, strides=2, padding='same'),
                      keras.layers.normalization.BatchNormalization(),
                      {'type': tf.nn.relu},

                      keras.layers.Conv2DTranspose(filters=128, kernel_size=ks, strides=2, padding='same'),
                      keras.layers.normalization.BatchNormalization(),
                      {'type': tf.nn.relu},

                      keras.layers.Conv2DTranspose(filters=32, kernel_size=ks, strides=2, padding='same'),
                      keras.layers.normalization.BatchNormalization(),
                      {'type': tf.nn.relu},

                      keras.layers.Conv2D(filters=1, kernel_size=ks, strides=1, padding='same'),
                      {'type': tf.reshape, 'shape': [-1, image_size, image_size]}
                    ]

    x = tf.placeholder(tf.float32, [None, image_size, image_size], name='x')
    y = tf.placeholder(tf.float32, proxy_size, name='y')
    z = tf.placeholder(tf.float32, [None, z_dim], name='z')
    model = VAE(layers_encode, layers_decode)

    loss = model.loss(x, y, distance_regularizer)

    tf.identity(model.samples(batch_size, z_dim), name = 'sample')
    tf.identity(model.encode(x).mu, name = 'encode')
    tf.identity(model.reconstruct(x), name = 'reconstruct')
    tf.identity(model.decode(z), name='decode')

    graph = tf.get_default_graph()

    def collect_updates(layers):
        update_ops = []
        for layer in layers:
            if isinstance(layer, keras.engine.Layer):
                update_ops += layer.updates
        return update_ops

    update_ops = collect_updates(layers_encode) + collect_updates(layers_decode)

    step = tf.train.AdamOptimizer(0.001).minimize(loss[0])

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)

    for ie in range(n_epochs):
        alpha = alphas[0] + (float(ie) / n_epochs) * (alphas[1] - alphas[0])
        beta = betas[0] + (float(ie) / n_epochs) * (betas[1] - betas[0])
        print("Epoch {:d}/{:d}".format(ie + 1, n_epochs))

        for ib, sample in enumerate(train_loader):
            images = sample[0]
            proxy_labels = sample[1]
            _, _, (_, output) = sess.run([step, update_ops, loss], { x: images, y: proxy_labels, z: np.zeros((1, z_dim), dtype=np.float32),
                                                                     model.alpha: alpha, model.beta: beta, K.learning_phase(): 1 })
            if (ib + 1) % 10 == 0:
                print("Batch {:d}: {:s}".format(ib + 1, format_log_string(output, log_fields)))

        val_output = evaluate(sess, x, y, loss, val_loader, log_fields, { model.alpha: 1.0, model.beta: 1.0,
                                                                          K.learning_phase(): 0 })
        print(">> Validation: {:s}".format(format_log_string(val_output, log_fields)))

    for sample in val_loader:
        images = sample[0]
        proxy_labels = sample[1]
        _, val_output = sess.run(loss, {x: images, y: proxy_labels, model.alpha: 1.0, model.beta: 1.0, K.learning_phase(): 0})
        samples = sess.run(graph.get_tensor_by_name('sample:0'), {model.alpha: 1.0, model.beta: 1.0, K.learning_phase(): 0})
        visualize = np.concatenate((images.reshape(-1, image_size), val_output['x_hat'].reshape(-1, image_size),
                                    samples.reshape(-1, image_size)), axis=1)
        break

    misc.imsave(save_dir + 'recons.png', visualize)
    saver.save(sess, save_dir + 'VAE')
    latent_expr(sess, model, val_data, save_dir)


    # misc.imsave('samples.png', visualize_grid(sess.run(x_samples, { K.learning_phase(): 0 })))


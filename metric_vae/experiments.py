import tensorflow as tf
import numpy as np
import os
from scipy import misc
import metric_vae.nn.model
from metric_vae.nn.model import VAE
import train
from load_data import DataLoader, load_tfd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tsne
import keras
import keras.backend as K

def latent_expr(sess, model, data, save_dir):
    encode = model['encode']
    alpha = model['alpha']
    beta = model['beta']
    x = model['x']
    images= data['images']
    lp = tf.get_default_graph().get_tensor_by_name('batch_normalization_1/keras_learning_phase:0')
    z = sess.run(encode, {x: images, alpha: 1.0, beta: 1.0, lp: 0})
    Y = tsne.tsne(z, 2, 50, 20.0)
    plt.scatter(Y[:, 0], Y[:, 1], c = data['labels'])
    plt.colorbar(ticks=np.arange(1, 7))
    plt.savefig(save_dir + 'latent_expr.png')
    plt.close()

def interpolate(sess, model, data, save_dir):
    encode = model['encode']
    decode = model['decode']
    alpha = model['alpha']
    beta = model['beta']
    x = model['x']
    z = model['z']
    images = data['images']
    labels = data['labels']
    lp = tf.get_default_graph().get_tensor_by_name('batch_normalization_1/keras_learning_phase:0')
    sad_faces = images[np.where(labels==3)][:20]
    happy_faces = images[np.where(labels==4)][:20]
    sad_z = sess.run(encode, {x: sad_faces, alpha: 1.0, beta: 1.0, lp: 0})
    happy_z = sess.run(encode, {x: happy_faces, alpha: 1.0, beta: 1.0, lp: 0})
    l = np.linspace(0, 1, 10)
    interpolations =  [(1 - t) * sad_z + t * happy_z for t in l[1:-1]]
    result = [sess.run(decode, {z: i, alpha: 1.0, beta: 1.0, lp: 0}).reshape(-1, images.shape[-1]) for i in interpolations]
    result = [sad_faces.reshape(-1, images.shape[-1])] + result + [happy_faces.reshape(-1, images.shape[-1])]
    result = np.concatenate(result, axis=1)
    misc.imsave(save_dir + 'interpolate.png', result)

def arithmetic(sess, model, data, save_dir):
    encode = model['encode']
    decode = model['decode']
    alpha = model['alpha']
    beta = model['beta']
    x = model['x']
    z = model['z']
    images = data['images']
    labels = data['labels']
    lp = tf.get_default_graph().get_tensor_by_name('batch_normalization_1/keras_learning_phase:0')

    zs = sess.run(encode, {x: images, alpha: 1.0, beta: 1.0, lp: 0})
    z_happy = zs[np.where(labels == 4)]
    z_neutral = zs[np.where(labels == 6)]
    z_sad = zs[np.where(labels == 3)]
    z_nothappy = zs[np.where(labels != 4)]
    
    happy_average = np.mean(z_happy, axis=0, keepdims=True)
    sad_average = np.mean(z_sad, axis=0, keepdims=True)
    nothappy_average = np.mean(z_nothappy, axis=0, keepdims=True)

    make_neutral_happy =  z_neutral[:20] - nothappy_average + happy_average
    make_sad_happy = z_sad[:20] - nothappy_average + happy_average
    make_sad_happy2 = z_sad[:20] - sad_average + happy_average

    make_neutral_happy = (sess.run(decode, {z: make_neutral_happy, alpha: 1.0, beta: 1.0, lp: 0})).reshape(-1, images.shape[-1])
    make_sad_happy = sess.run(decode, {z: make_sad_happy, alpha: 1.0, beta: 1.0, lp: 0}).reshape(-1, images.shape[-1])
    make_sad_happy2 = sess.run(decode, {z: make_sad_happy2, alpha: 1.0, beta: 1.0, lp: 0}).reshape(-1, images.shape[-1])

    make_neutral_happy = np.concatenate([images[np.where(labels == 6)][:20].reshape(-1, images.shape[-1]), make_neutral_happy], axis=1)
    make_sad_happy = np.concatenate([images[np.where(labels == 3)][:20].reshape(-1, images.shape[-1]), make_sad_happy], axis=1)
    make_sad_happy2 = np.concatenate([images[np.where(labels == 3)][:20].reshape(-1, images.shape[-1]), make_sad_happy2], axis=1)

    misc.imsave(save_dir + 'make_neutral_sad.png', make_neutral_happy)
    misc.imsave(save_dir + 'make_happy_sad.png', make_sad_happy)
    misc.imsave(save_dir + 'make_happy_sad2.png', make_sad_happy2)


if __name__ == '__main__':
    train_data, val_data = load_tfd()
    for i in reversed(range(100)):
        save_dir = 'experiments/' + str(i) + '/'
        if os.path.exists(save_dir):
            break

    saver = tf.train.import_meta_graph(save_dir + 'VAE.meta')
    sess = tf.Session()
    saver.restore(sess, save_dir + 'VAE')

    graph = tf.get_default_graph()
    model = {'reconstruct': graph.get_tensor_by_name('reconstruct:0'),
             'sample': graph.get_tensor_by_name('sample:0'),
             'encode': graph.get_tensor_by_name('encode:0'),
             'decode': graph.get_tensor_by_name('decode:0'),
             'alpha': graph.get_tensor_by_name('alpha:0'),
             'beta': graph.get_tensor_by_name('beta:0'),
             'x': graph.get_tensor_by_name('x:0'),
             'z': graph.get_tensor_by_name('z:0')
            }

    latent_expr(sess, model, val_data, save_dir)
    interpolate(sess, model, val_data, save_dir)
    arithmetic(sess, model, val_data, save_dir)

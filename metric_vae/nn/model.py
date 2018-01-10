import tensorflow as tf
import numpy as np
import os

class VAE(object):
    def __init__(self, layers_encode, layers_decode):
        self.encoder = Encoder(layers_encode)
        self.decoder = Decoder(layers_decode)
        self.alpha = tf.placeholder(tf.float32, name='alpha')
        self.beta = tf.placeholder(tf.float32, name='beta')
        self.tolerance = 0.1

    def encode(self, x):
        return self.encoder.encode(x)

    def decode(self, z):
        return self.decoder.decode(z)

    def loss(self, x, proxy_label, distance_regularizer):
        q_z = self.encode(x)
        z = q_z.sample()
        x_hat = self.decode(z)

        # if proxy_label != 'none':
        #     proxy = proxy_label
        # else:
        #     proxy = x
        # if tf.rank(proxy_label)==2:
        #     proxy = proxy_label
        # else:
        #     distance_regularizer = 'none'
        proxy = proxy_label


        if distance_regularizer == 'none':
            dr = tf.constant(0.)
        elif distance_regularizer == 'pointwise':
            dr = tf.reduce_mean(tf.abs(tf.reduce_sum(tf.square(proxy), axis=1) - tf.reduce_sum(tf.square(z), axis=1)))
        elif distance_regularizer == 'expanding_pointwise':
            dr = tf.reduce_mean(
                tf.maximum(tf.reduce_sum(tf.square(proxy), axis=1) - tf.reduce_sum(tf.square(z), axis=1) + self.tolerance, 0))
        elif distance_regularizer == 'contracting_pointwise':
            dr = tf.reduce_mean(
                tf.maximum(tf.reduce_sum(tf.square(z), axis=1) - tf.reduce_sum(tf.square(proxy), axis=1) + self.tolerance, 0))
        elif distance_regularizer == 'batchwise':
            points = tf.concat([z, proxy], axis=1)
            a = tf.expand_dims(points, 0)
            b = tf.expand_dims(points, 1)
            y = a - b
            y = tf.reshape(y, [-1, tf.shape(z)[1] + tf.shape(proxy)[1]])
            d1 = tf.reduce_sum(tf.square(y[:, :tf.shape(z)[1]]), axis=1)
            d2 = tf.reduce_sum(tf.square(y[:, tf.shape(z)[1]:]), axis=1)
            m1, v1 = tf.nn.moments(d1, axes=[0])
            m2, v2 = tf.nn.moments(d2, axes=[0])
            dr = tf.abs((d1 - m1)/v1 - (d2 - m2)/v2)
            dr = tf.reduce_mean(dr)
        else:
            raise ValueError('unknown distance regularizer: ' + distance_regularizer)

        recon_val = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x_hat, x), axis=[1,2]))
        kl_val = tf.reduce_mean(q_z.kl_div())

        # loss_val = recon_val + kl_val * self.alpha + dr * self.beta
        # loss_val = kl_val * self.alpha + dr * self.beta
        loss_val = recon_val + kl_val * self.alpha + dr * self.beta

        return loss_val, {
            'loss': loss_val,
            'recon': recon_val,
            'kl': kl_val,
            'dr': dr,
            'x_hat': x_hat,
            'z': z
        }

    def reconstruct(self, x):
        q_z = self.encode(x)
        z = q_z.sample()
        return self.decode(z)

    def samples(self, n_samples, z_dim):
        return self.decode(tf.random_normal([n_samples, z_dim]))

def to_layer(layer):
    if isinstance(layer, dict):
        args = {n: layer[n] for n in layer if n != 'type'}
        return lambda x: layer['type'](x, **args)
    else:
        return layer

def apply_layers(layers, x):
    ret = x
    for layer in layers:
        ret = to_layer(layer)(ret)

    return ret

class GaussianDist(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        eps = tf.random_normal(tf.shape(self.mu))
        return self.mu + self.sigma * eps

    def kl_div(self):
        sigma_squared = tf.pow(self.sigma, 2)
        return -0.5 * tf.reduce_sum(1.0 + tf.log(sigma_squared) - tf.pow(self.mu, 2) - sigma_squared, axis=1)

class Encoder(object):
    def __init__(self, layers):
        self.layers = layers

    def encode(self, x):
        ret = apply_layers(self.layers, x)

        mu = ret[:,:,0]
        sigma = tf.clip_by_value(tf.nn.softplus(ret[:,:,1]), 1e-5, np.inf)
        # sigma = 0.5
        return GaussianDist(mu, sigma)

class Decoder(object):
    def __init__(self, layers):
        self.layers = layers

    def decode(self, z):
        return apply_layers(self.layers, z)

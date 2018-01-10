import tensorflow as tf
import numpy as np
import os

class Classifier():

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, penultimate=False):
        h, z, softz = apply_layers(self.layers, x)
        if penultimate: return h, z, softz
        else: return softz

    def loss(self, y_, x):
        mse = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_, self.forward(x)), axis=1))
        return mse


def to_layer(layer):
    if isinstance(layer, dict):
        args = {n: layer[n] for n in layer if n != 'type'}
        return lambda x: layer['type'](x, **args)
    else:
        return layer

def apply_layers(layers, x):
    h = x
    for layer in layers[:-1]:
        h = to_layer(layer)(h)
    z = to_layer(layers[-1])(h)

    return h, z, tf.nn.softmax(z)










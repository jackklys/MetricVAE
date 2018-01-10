import tensorflow as tf
import numpy as np
import os


def conv_and_pool(x, num_outputs, conv_size, pool_size, activation_fn=None):
    x = tf.contrib.layers.conv2d(x, num_outputs=num_outputs, kernel_size=conv_size, activation_fn=activation_fn)
    x = tf.contrib.layers.max_pool2d(x, kernel_size=pool_size)
    return x

# def dense_and_dropout(x, units, rate, activation=tf.nn.relu, training=False):
#     x = tf.layers.dense(x, units=units, activation=activation)
#     x = tf.layers.dropout(x, rate=rate, training=training )
#     return x

def linear(x, n_in, n_out):
    W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[n_out]))
    return tf.matmul(x, W) + b

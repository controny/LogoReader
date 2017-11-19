# coding: utf-8

import tensorflow as tf

def inference(input_tensor, input_tensor_size, num_labels):
    weights = tf.Variable(tf.truncated_normal([input_tensor_size, num_labels], stddev=0.001))
    biases = tf.Variable(tf.zeros([num_labels]))
    logit = tf.matmul(input_tensor, weights) + biases

    return logit

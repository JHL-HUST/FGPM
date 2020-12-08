# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
from Config import config

def textbirnn(input_x, dropout_keep_prob, dataset, reuse=False):
    """
    A Bi-directional RNN for text classification.
    Uses an embedding layer, followed by a bi-directional LSTM layer, a dropout layer and a fully-connected layer.
    """

    num_classes = config.num_classes[dataset]
    vocab_size = config.num_words[dataset]
    embedding_size = 300

    # Embedding layer
    with tf.variable_scope("embedding", reuse=reuse):
        embeddings = tf.get_variable("W", 
            initializer=tf.random_uniform([vocab_size+1, embedding_size], -1.0, 1.0),
            trainable=True)
        embedded_chars = tf.nn.embedding_lookup(embeddings, input_x, name="embedded_chars")  # [None, sequence_length, embedding_size]

    # Create a bi-directional LSTM layer for each rnn layer
    with tf.variable_scope('bilstm', reuse=reuse):
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell
        
        def get_bi_cell():
            fw_cell = cell_fun(128, state_is_tuple=True) #forward direction cell
            bw_cell = cell_fun(128, state_is_tuple=True) #backward direction cell
            return fw_cell, bw_cell

        # Bi-lstm layer
        fw_cell, bw_cell = get_bi_cell()
        outputs, last_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_chars, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2)
        output = tf.reduce_mean(outputs, axis=1)

    # Add dropout
    with tf.variable_scope("dropout", reuse=reuse):
        rnn_drop = tf.nn.dropout(output, dropout_keep_prob)

    # Final (unnormalized) scores and predictions
    with tf.variable_scope("output", reuse=reuse):
        W = tf.get_variable(
            "W",
            shape=[128*2, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", initializer=tf.constant(0.1, shape=[num_classes]))

        scores = tf.nn.xw_plus_b(rnn_drop, W, b, name="scores")
        predictions = tf.argmax(scores, 1, name="predictions", output_type=tf.int32)

    return embeddings, embedded_chars, predictions, scores
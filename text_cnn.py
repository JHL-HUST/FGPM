# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from Config import config


def textcnn(input_x, dropout_keep_prob, dataset, reuse=False):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by three convolutional + max-pooling layers, a dropout layer and a fully-connected layer.
    """
    sequence_length = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    vocab_size = config.num_words[dataset]

    embedding_size = 300
    filter_sizes = [3, 4, 5]
    num_filters = 128

    with tf.variable_scope("test", reuse=reuse):
        # Embedding layer
        with tf.variable_scope("embedding", reuse=reuse):
            embeddings = tf.get_variable(
                initializer=tf.random_uniform([vocab_size + 1, embedding_size], -1.0, 1.0),
                name="W",
                trainable=True,
            )
            embedded_chars = tf.nn.embedding_lookup(
                embeddings, input_x, name="embedded_chars"
            )  # [None, sequence_length, embedding_size]
            embedded_chars_expanded = tf.expand_dims(
                embedded_chars, -1
            )  # [None, sequence_length, embedding_size, 1]

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=reuse):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.get_variable(
                    initializer=tf.truncated_normal(filter_shape, stddev=0.1),
                    name="W",
                )
                b = tf.get_variable(
                    initializer=tf.constant(0.1, shape=[num_filters]), name="b"
                )
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv",
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool",
                )
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.variable_scope("dropout", reuse=reuse):
            h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob, name="text_vector")

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output", reuse=reuse):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            b = tf.get_variable(initializer=tf.constant(0.1, shape=[num_classes]), name="b")
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            predictions = tf.argmax(scores, 1, name="predictions", output_type=tf.int32)

    return embeddings, embedded_chars, predictions, scores


def compute_loss(logits, input_y, num_classes):
    losses = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(input_y, depth=num_classes), logits=logits
        )
    )
    return losses


def compute_acc(predictions, input_y):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(input_y, predictions), tf.float32))
    return accuracy

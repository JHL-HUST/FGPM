# -*- coding: UTF-8 -*-
"""
Normal training & adversarial training with FGPM.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import numpy as np
import time
import tensorflow as tf
from text_cnn import textcnn, compute_loss, compute_acc
from text_rnn import textrnn
from text_birnn import textbirnn
from FGPM import FGPM
from utils import text_encoder, read_text, load_dictionary, generate_model_save_path
from Config import config

# Dataset params
tf.flags.DEFINE_string(
    "data", "ag_news", "Dataset (choices: dbpedia, yahoo_answers, ag_news)"
)
# Model and training params
tf.flags.DEFINE_string(
    "nn_type", "textcnn", "The neural network classification model (choices: textcnn, textrnn, textbirnn)"
)
tf.flags.DEFINE_string(
    "train_type", "org", "Normal train or adversarial train (choices: org, adv)"
)
tf.flags.DEFINE_float(
    "adv_sigma",
    0.5,
    "Hypermeter combining original and adversarial loss when adv-training",
)
tf.flags.DEFINE_float(
    "regularization_coef",
    0.5,
    "Hypermeter combining adversarial loss and regularization term when adv-training",
)
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 3, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer(
    "num_checkpoints", 2, "Number of checkpoints to store (default: 5)"
)

# FGPM and attack parameters
tf.flags.DEFINE_integer("grad_upd_interval", 5, "grad update interval")
tf.flags.DEFINE_integer(
    "max_iter", 20, "Maximum number of substitutions allowed.(default: 20)"
)
tf.flags.DEFINE_float(
    "distance_threshold",
    0.5,
    "The maximum distance between two substitutions (default: 0.5)",
)
tf.flags.DEFINE_integer(
    "max_candidates",
    20,
    "Maximum number of substitution candidates per word. (default: 20)",
)

# GPU params
tf.flags.DEFINE_string("gpu", "0", "gpu to use")
tf.flags.DEFINE_boolean(
    "allow_soft_placement", True, "Allow device soft device placement"
)
tf.flags.DEFINE_boolean(
    "log_device_placement", False, "Log placement of ops on devices"
)

# File path params
tf.flags.DEFINE_string(
    "data_dir", "./", "The path to hold the input data",
)
tf.flags.DEFINE_string(
    "model_dir", "./", "The path to hold the output data",
)

FLAGS = tf.flags.FLAGS

MAX_VOCAB_SIZE = MAX_VOCAB_SIZE = config.num_words[FLAGS.data]
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


def train(
    train_seqs,
    train_seqs_mask,
    train_labels,
    test_seqs,
    test_seqs_mask,
    test_labels,
    embedding_matrix,
    dist_mat,
    num_classes,
):

    num_examples = len(train_labels)
    with tf.Graph().as_default():
        # Construct calculation graph
        dist_mat_tensor = tf.constant(dist_mat[:, : FLAGS.max_candidates, :])

        global_step = tf.Variable(0, dtype=tf.int64)
        step_update = global_step.assign_add(1)

        x = tf.placeholder(
            tf.int32, shape=[None, config.word_max_len[FLAGS.data]]
        )
        x_mask = tf.placeholder(
            tf.int32, shape=[None, config.word_max_len[FLAGS.data]]
        )
        y = tf.placeholder(tf.int32, shape=[None])

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((x, x_mask, y))
            .shuffle(num_examples)
            .batch(FLAGS.batch_size)
            .prefetch(buffer_size=1)
        )
        test_dataset = (
            tf.data.Dataset.from_tensor_slices((x, x_mask, y))
            .shuffle(num_examples)
            .batch(200)
            .prefetch(buffer_size=1)
        )

        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types, train_dataset.output_shapes
        )

        x_batch, x_mask_batch, y_batch = iterator.get_next()
        dropout_keep_prob = tf.Variable(1.0, trainable=False, dtype=tf.float32)

        train_initializer = iterator.make_initializer(train_dataset)
        test_initializer = iterator.make_initializer(test_dataset)

        if FLAGS.nn_type == "textcnn":
            embeddings, _, predictions, logits = textcnn(
                x_batch, dropout_keep_prob, FLAGS.data
            )
        elif FLAGS.nn_type == "textrnn":
            embeddings, _, predictions, logits = textrnn(
                x_batch, dropout_keep_prob, FLAGS.data
            )
        elif FLAGS.nn_type == "textbirnn":
            embeddings, _, predictions, logits = textbirnn(
                x_batch, dropout_keep_prob, FLAGS.data
            )

        loss_normal = compute_loss(logits, y_batch, num_classes)
        acc_normal = compute_acc(predictions, y_batch)

        tf.summary.scalar("natual-loss", loss_normal)
        tf.summary.scalar("natual-acc", acc_normal)

        if FLAGS.train_type == "adv":
            x_adv, _, _ = FGPM(
                x_batch,
                y_batch,
                x_mask_batch,
                FLAGS.data,
                FLAGS.nn_type,
                FLAGS.max_iter,
                num_classes,
                dist_mat_tensor,
                FLAGS.grad_upd_interval,
                sn=FLAGS.max_candidates,
            )

            if FLAGS.nn_type == "textcnn":
                _, _, predictions_adv, logits_adv = textcnn(
                    x_adv, dropout_keep_prob, FLAGS.data, reuse=True
                )
            elif FLAGS.nn_type == "textrnn":
                _, _, predictions_adv, logits_adv = textrnn(
                    x_adv, dropout_keep_prob, FLAGS.data, reuse=True
                )
            elif FLAGS.nn_type == "textbirnn":
                _, _, predictions_adv, logits_adv = textbirnn(
                    x_adv, dropout_keep_prob, FLAGS.data, reuse=True
                )

            loss_adversarial = compute_loss(logits_adv, y_batch, num_classes)
            acc_adversarial = compute_acc(predictions_adv, y_batch)

            tf.summary.scalar("adversarial-loss", loss_adversarial)

            standard_loss = (
                FLAGS.adv_sigma * loss_normal
                + (1 - FLAGS.adv_sigma) * loss_adversarial
            )
            pair_logits_loss = tf.losses.mean_squared_error(
                logits_adv,
                logits,
                weights=FLAGS.regularization_coef,
                reduction=tf.losses.Reduction.MEAN,
            )
            tf.summary.scalar("standard_loss", standard_loss)
            tf.summary.scalar("pair_logits_loss", pair_logits_loss)
            loss = standard_loss + pair_logits_loss
        elif FLAGS.train_type == "org":
            loss = loss_normal
            acc_adversarial = acc_normal
        else:
            raise NotImplementedError

        tf.summary.scalar("adversarial-acc", acc_adversarial)
        tf.summary.scalar("loss", loss)

        tvars = tf.trainable_variables()
        opt = tf.train.AdamOptimizer().minimize(loss, var_list=tvars)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
            os.path.join(
                FLAGS.model_dir,
                "runs_%s" % FLAGS.nn_type,
                generate_model_save_path(timestamp, FLAGS.data, FLAGS.train_type),
            )
        )
        print("Saving model to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=FLAGS.num_checkpoints
        )

        session_conf = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=session_conf))

        logging_str = ""

        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            sess.run(tf.assign(embeddings, embedding_matrix.T))

            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(out_dir)

            start_train_time = time.time()
            for epoch in range(FLAGS.num_epochs):
                # train the model, drop_keep_rate = 0.8
                tf.assign(dropout_keep_prob, 0.8)
                sess.run(
                    train_initializer,
                    feed_dict={x: train_seqs, x_mask: train_seqs_mask, y: train_labels},
                )
                total_train_acc_normal = 0
                total_train_acc_adversarial = 0
                no_train_examples = len(train_labels)
                try:
                    while True:
                        (
                            temp_train_acc_normal,
                            temp_train_acc_adversarial,
                            step,
                            summary,
                            _,
                        ) = sess.run(
                            [
                                acc_normal,
                                acc_adversarial,
                                step_update,
                                merged_summary,
                                opt,
                            ]
                        )
                        train_writer.add_summary(summary, step)
                        total_train_acc_normal += (
                            temp_train_acc_normal * FLAGS.batch_size
                        )
                        total_train_acc_adversarial += (
                            temp_train_acc_adversarial * FLAGS.batch_size
                        )
                except tf.errors.OutOfRangeError:
                    pass
                # validate the model, drop_keep_rate = 1.0
                tf.assign(dropout_keep_prob, 1.0)
                sess.run(
                    test_initializer,
                    feed_dict={x: test_seqs, x_mask: test_seqs_mask, y: test_labels},
                )
                total_test_acc = 0
                no_test_examples = len(test_labels)
                try:
                    while True:
                        temp_test_acc = sess.run(acc_normal)
                        total_test_acc += temp_test_acc * 200
                except tf.errors.OutOfRangeError:
                    pass

                logging_str = (
                    logging_str
                    + "Epoch {}\n".format(str(epoch + 1))
                    + "---------------------------\n"
                    + "Training normal accuracy is {}\n".format(
                        total_train_acc_normal / no_train_examples
                    )
                    + "Training adversarial accuracy is {}\n".format(
                        total_train_acc_adversarial / no_train_examples
                    )
                    + "Validation accuracy is {}\n".format(
                        total_test_acc / no_test_examples
                    )
                    + "---------------------------\n"
                )

                print(logging_str)

                saver.save(
                    sess, checkpoint_prefix + "_" + str(epoch + 1),
                )

            end_train_time = time.time()
            train_writer.close()
            logging_str += "Training Time: {}\n".format(
                end_train_time - start_train_time
            )

    # output flags log
    flags_log = ""
    for name, value in FLAGS.__flags.items():
        flags_log += str(name) + ":\t" + str(value.value) + "\n"

    log_save_path = os.path.join(out_dir, "log.txt")
    with open(log_save_path, "a", encoding="utf-8") as f:
        f.write(time.strftime("\n%Y-%m-%d %H:%M:%S\n", time.localtime(time.time())))
        f.write(flags_log + logging_str)


def main(argv=None):

    org_dic, _ = load_dictionary(FLAGS.data, MAX_VOCAB_SIZE, FLAGS.data_dir)
    train_texts, train_labels = read_text("%s/train" % FLAGS.data, FLAGS.data_dir)
    test_texts, test_labels = read_text("%s/test" % FLAGS.data, FLAGS.data_dir)

    train_seqs, train_seqs_mask = text_encoder(
        train_texts, org_dic, config.word_max_len[FLAGS.data]
    )
    test_seqs, test_seqs_mask = text_encoder(
        test_texts, org_dic, config.word_max_len[FLAGS.data]
    )
    print("Dataset ", FLAGS.data, " loaded!")
    glove_embedding_matrix = np.load(
        FLAGS.data_dir
        + "aux_files/embeddings_glove_%s_%d.npy" % (FLAGS.data, MAX_VOCAB_SIZE)
    )
    dist_mat = np.load(
        FLAGS.data_dir
        + "aux_files/small_dist_counter_%s_%d.npy" % (FLAGS.data, MAX_VOCAB_SIZE)
    )
    for stop_word in config.stop_words:
        if stop_word in org_dic:
            dist_mat[org_dic[stop_word], :, :] = 0

    train_log = train(
        train_seqs,
        train_seqs_mask,
        train_labels,
        test_seqs,
        test_seqs_mask,
        test_labels,
        glove_embedding_matrix,
        dist_mat,
        config.num_classes[FLAGS.data],
    )


if __name__ == "__main__":
    tf.app.run()

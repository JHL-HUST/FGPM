import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf
from utils import (
    load_dist_mat,
    read_text,
    text_encoder,
    load_dictionary,
    generate_model_save_path,
    calculate_diff,
    calculate_diff_for_array,
)
import numpy as np
from text_cnn import textcnn, compute_acc, compute_loss
from text_rnn import textrnn
from text_birnn import textbirnn
from FGPM import FGPM
from Config import config
import pickle
import time
import math

# Dataset params
tf.flags.DEFINE_string(
    "data", "ag_news", "Dataset (dbpedia, yahoo_answers, ag_news)"
)

# Model loading params
tf.flags.DEFINE_string(
    "nn_type", "textcnn", "The neural network classification model (choices: textcnn, textrnn, textbirnn)"
)
tf.flags.DEFINE_string(
    "train_type", "org", "The training way of the model to be loaded (choices: org, adv)"
)
tf.flags.DEFINE_string("time", None, "The timestamp of the model to be loaded")
tf.flags.DEFINE_string("step", None, "The checkpoint epoch of the model to be loaded")

# Attack params
tf.flags.DEFINE_integer("batch_size", 200, "The number of randomly selected samples to be attacked (default: 200)")
tf.flags.DEFINE_string("recipe", "FGPM", "The attack recipe (default: FGPM)")
tf.flags.DEFINE_boolean(
    "evaluate_testset", True, "Evaluate the entire test set before attack."
)
tf.flags.DEFINE_boolean(
    "stop_words", True, "Do not modify stop words, such as prepositions and articles."
)
tf.flags.DEFINE_boolean(
    "save_to_file",
    True,
    "Save adverarial examples and attack results to file <project-dir>/adv_samples/~.",
)

# Synonyms params
tf.flags.DEFINE_float(
    "distance_threshold",
    0.5,
    "The maximum distance between two substitutions (default: 0.5)",
)
tf.flags.DEFINE_integer(
    "max_candidates",
    4,
    "Use the nearest `max_candidates` synonyms that meet the delta constraint when attacking (default: 4)",
)

# FGPM params
tf.flags.DEFINE_integer("max_iter", 30, "Maximum number of substitutions allowed.")
tf.flags.DEFINE_integer("grad_upd_interval", 1, "grad update interval")
tf.flags.DEFINE_float(
    "max_perturbed_percent",
    0.25,
    "Upper bound for word substitution ratio (default: 0.25)",
)


# GPU params
tf.flags.DEFINE_string("gpu", "0", "GPU to use (default: 0)")

# File path params
tf.flags.DEFINE_string(
    "data_dir", "./", "The path to hold the input data",
)
tf.flags.DEFINE_string(
    "model_dir", "./", "The path to hold the output data",
)

FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
MAX_VOCAB_SIZE = config.num_words[FLAGS.data]


def generate_model_path(model_dir):
    CHECKPOINT_DIR = os.path.join(
        model_dir,
        "./runs_%s/%s/checkpoints/"
        % (
            FLAGS.nn_type,
            generate_model_save_path(FLAGS.time, FLAGS.data, FLAGS.train_type),
        ),
    )
    if FLAGS.step == "":
        checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    else:
        checkpoint_file = CHECKPOINT_DIR + "model_%s" % (FLAGS.step)
    print(checkpoint_file)
    return checkpoint_file


def sample(clean_text_list, labels, sample_num):
    """
    Use the Numpy library to randomly select the samples to be attacked. 
    Note that the seed used in our experiments is 0.
    """
    clean_text_list = np.array(clean_text_list)
    labels = np.array(labels)
    np.random.seed(0)
    shuffled_idx = np.arange(0, len(clean_text_list), 1)
    np.random.shuffle(shuffled_idx)
    sampled_idx = shuffled_idx[:sample_num]
    return list(clean_text_list[sampled_idx]), list(labels[sampled_idx])


def encode_convert_to_text(
    perturbed_encoded_text, sample_encoded_text, sample_clean_text, org_inv_dic, dataset
):

    index_overflow = False
    ori_tokens = sample_clean_text.split()
    perturbed_tokens = ori_tokens.copy()
    for i in range(min(len(ori_tokens), config.word_max_len[dataset])):
        if perturbed_encoded_text[i] != sample_encoded_text[i]:
            if perturbed_encoded_text[i] == -1 or perturbed_encoded_text[i] == 0:
                index_overflow = True
                continue
            perturbed_tokens[i] = org_inv_dic[perturbed_encoded_text[i]]
    return index_overflow, " ".join(perturbed_tokens)


def check_index_overflow(
    perturbed_encoded_text, sample_encoded_text, sample_clean_text, dataset
):
    index_overflow = False
    ori_tokens = sample_clean_text.split()
    for i in range(min(len(ori_tokens), config.word_max_len[dataset])):
        if perturbed_encoded_text[i] != sample_encoded_text[i]:
            if perturbed_encoded_text[i] <= 0:
                index_overflow = True
                break
    return index_overflow


def output_flags_log(FLAGS):
    flags_log = ""
    for name, value in FLAGS.__flags.items():
        flags_log += str(name) + ":\t" + str(value.value) + "\n"
    return flags_log

def main(argv=None):

    tf.reset_default_graph()
    session_conf = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=session_conf)
    )

    x = tf.placeholder(
        tf.int32, shape=[None, config.word_max_len[FLAGS.data]]
    )
    x_mask = tf.placeholder(
        tf.int32, shape=[None, config.word_max_len[FLAGS.data]]
    )
    y = tf.placeholder(tf.int32, shape=[None])
    x_org = tf.placeholder(
        tf.int32, shape=[None, config.word_max_len[FLAGS.data]]
    )

    if FLAGS.nn_type == "textcnn":
        _, embedded_chars, predictions, scores = textcnn(x, 1.0, FLAGS.data)
    elif FLAGS.nn_type == "textrnn":
        _, embedded_chars, predictions, scores = textrnn(x, 1.0, FLAGS.data)
    elif FLAGS.nn_type == "textbirnn":
        _, embedded_chars, predictions, scores = textbirnn(x, 1.0, FLAGS.data)

    saver = tf.train.Saver()
    checkpoint_file = generate_model_path(FLAGS.model_dir)
    saver.restore(sess, checkpoint_file)

    org_dic, org_inv_dic = load_dictionary(FLAGS.data, MAX_VOCAB_SIZE, FLAGS.data_dir)
    print("The dictionary has %d words." % len(org_dic))
    if FLAGS.train_type == "org" or FLAGS.train_type == "adv":
        dist_mat = load_dist_mat(FLAGS.data, MAX_VOCAB_SIZE, FLAGS.data_dir)
    else:
        raise NotImplementedError
    if FLAGS.stop_words:
        print("Enable stop words.")
        for stop_word in config.stop_words:
            if stop_word in org_dic:
                dist_mat[org_dic[stop_word], :, :] = 0
    dist_mat = dist_mat[:, : FLAGS.max_candidates, :]
    clean_texts, labels = read_text("%s/test" % FLAGS.data, data_dir=FLAGS.data_dir)
    encoded_texts, _ = text_encoder(
        clean_texts, org_dic, config.word_max_len[FLAGS.data]
    )

    if FLAGS.evaluate_testset:
        print("Model accuracy on test set:")
        correct_predict_count = 0
        sample_num = len(clean_texts)
        for i in range(math.ceil(sample_num / 500)):
            pred = sess.run(
                predictions,
                feed_dict={
                    x: encoded_texts[i * 500 : (i + 1) * 500],
                    y: labels[i * 500 : (i + 1) * 500],
                },
            )
            for j in range(len(pred)):
                if pred[j] == labels[i * 500 + j]:
                    correct_predict_count += 1
        testset_acc = correct_predict_count / sample_num
        print(
            correct_predict_count, "/", sample_num, "=", testset_acc,
        )

    print("Sample ", FLAGS.batch_size, "samples to attack...")
    sample_clean_texts, sample_labels = sample(clean_texts, labels, FLAGS.batch_size)
    sample_encoded_texts, sample_encoded_texts_mask = text_encoder(
        sample_clean_texts, org_dic, config.word_max_len[FLAGS.data]
    )


    start_attack_time = time.time()

    # return [perturbed_encoded_texts, wrong_predict_state, res_adv_labels]
    if FLAGS.recipe == "FGPM":
        perturbed_x, wrong_predict_state_tensor, adv_labels = FGPM(
            x,
            y,
            x_mask,
            FLAGS.data,
            FLAGS.nn_type,
            FLAGS.max_iter,
            config.num_classes[FLAGS.data],
            dist_mat,
            FLAGS.grad_upd_interval,
            dis_threshold=FLAGS.distance_threshold,
            sn=FLAGS.max_candidates,
            max_perturbed_percent=FLAGS.max_perturbed_percent,
            xs_org=x_org,
        )

        print("FGPM Attack: Computation graph created!")

        perturbed_encoded_texts = []
        wrong_predict_state = []

        res_x, res_state, res_adv_labels = sess.run(
            [perturbed_x, wrong_predict_state_tensor, adv_labels],
            feed_dict={
                x: sample_encoded_texts[: FLAGS.batch_size],
                y: sample_labels[: FLAGS.batch_size],
                x_mask: sample_encoded_texts_mask[: FLAGS.batch_size],
                x_org: sample_encoded_texts[: FLAGS.batch_size],
            },
        )

        perturbed_encoded_texts = res_x
        wrong_predict_state = res_state
    else:
        raise NotImplementedError

    end_attack_time = time.time()

    save_path = None
    if FLAGS.save_to_file:
        if not os.path.exists('adv_samples'):
            os.makedirs('adv_samples')
        current_time = str(int(time.time()))
        save_path = "adv_samples/{}-{}-{}-{}-{}-{}.txt".format(
                FLAGS.recipe,
                FLAGS.data,
                FLAGS.nn_type,
                FLAGS.train_type,
                FLAGS.time,
                FLAGS.step,
            )
        save_file = open(save_path, "a", encoding="utf-8")
        save_file.write(output_flags_log(FLAGS))

    substitution_ratio = []
    unchanged_sample_count = 0
    success_attack_count = 0
    fail_count = 0

    sess.close()

    result_info = ""
    for j, perturbed_encoded_text in enumerate(perturbed_encoded_texts):

        index_overflow, adv_text = encode_convert_to_text(
            perturbed_encoded_text,
            sample_encoded_texts[j],
            sample_clean_texts[j],
            org_inv_dic,
            FLAGS.data,
        )

        if not wrong_predict_state[j]:
            fail_count += 1
        elif index_overflow:
            fail_count += 1
        else:
            diff = calculate_diff(sample_clean_texts[j], adv_text)
            if diff == 0:
                unchanged_sample_count += 1
            elif (
                diff / len(sample_clean_texts[j].split()) > FLAGS.max_perturbed_percent
            ):
                fail_count += 1
            else:
                success_attack_count += 1
                substitution_ratio.append(diff / len(sample_clean_texts[j].split()))

        log_info = (
            str(j)
            + "\noriginal text: "
            + sample_clean_texts[j]
            + "\noriginal label: "
            + str(sample_labels[j])
            + "\nperturbed text: "
            + adv_text
            + "\nperturbed label: "
            + str(res_adv_labels[j])
            + "\n"
        )
        if FLAGS.save_to_file:
            save_file.write(log_info)   

    model_acc_before_attack = 1.0 - unchanged_sample_count / FLAGS.batch_size
    model_acc_after_attack = (
        1.0 - (unchanged_sample_count + success_attack_count) / FLAGS.batch_size
    )

    if len(substitution_ratio) == 0:
        average_sub_ratio = 0.0
    else:
        average_sub_ratio = sum(substitution_ratio) / len(substitution_ratio)
    summary_table_rows = [
        ["ITEM", "VALUE"],
        ["Total Time For Attack:", end_attack_time - start_attack_time],
        ["Model Accuracy of Test Set:", testset_acc],
        ["Model Accuracy Before Attack:", model_acc_before_attack,],
        [
            "Attack Success Rate:",
            success_attack_count / (FLAGS.batch_size - unchanged_sample_count),
        ],
        ["Model Accuracy After Attack:", model_acc_after_attack,],
        ["Average Substitution Ratio:", average_sub_ratio,],
    ]
    for row in summary_table_rows:
        result_info += str(row[0]) + str(row[1]) + "\n"
    print(result_info)

    if FLAGS.save_to_file:
        save_file.write(result_info)
        save_file.close()

if __name__ == "__main__":
    tf.app.run()

import numpy as np
import pickle
import os
import re
import tensorflow as tf
import utils
from Config import config


tf.flags.DEFINE_string("data", "ag_news", "Dataset (dbpedia, yahoo_answers, ag_news)")
tf.flags.DEFINE_float("distance_threshold", 0.5, "Distance threshold between synonyms") 
tf.flags.DEFINE_string("data_dir", "./", "The path to hold the input data")
tf.flags.DEFINE_string("model_dir", "./", "The path to hold the output data")
tf.flags.DEFINE_string("gpu", "0", "Device index")

FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

MAX_VOCAB_SIZE = config.num_words[FLAGS.data]

if not os.path.exists(os.path.join(FLAGS.data_dir, 'aux_files')):
    os.makedirs(os.path.join(FLAGS.data_dir, 'aux_files'))

def main(argv=None):
    # Generate dictionary
    if not os.path.isfile(os.path.join(FLAGS.data_dir, 'aux_files', 'org_dic_%s_%d.pkl' % (FLAGS.data, MAX_VOCAB_SIZE))):
        print('org_dic & org_inv_dic not exist, build and save the dict...')
        org_dic, org_inv_dic, _ = utils.build_dict(FLAGS.data, MAX_VOCAB_SIZE, data_dir=FLAGS.data_dir)
        with open(os.path.join(FLAGS.data_dir, 'aux_files', 'org_dic_%s_%d.pkl' % (FLAGS.data, MAX_VOCAB_SIZE)), 'wb') as f:
            pickle.dump(org_dic, f, protocol=4)
        with open(os.path.join(FLAGS.data_dir, 'aux_files', 'org_inv_dic_%s_%d.pkl' % (FLAGS.data, MAX_VOCAB_SIZE)), 'wb') as f:
            pickle.dump(org_inv_dic, f, protocol=4)
    else:
        print('org_dic & org_inv_dic already exist, load the dict...')
        with open(os.path.join(FLAGS.data_dir, 'aux_files', 'org_dic_%s_%d.pkl' % (FLAGS.data, MAX_VOCAB_SIZE)), 'rb') as f:
            org_dic = pickle.load(f)
        with open(os.path.join(FLAGS.data_dir, 'aux_files', 'org_inv_dic_%s_%d.pkl' % (FLAGS.data, MAX_VOCAB_SIZE)), 'rb') as f:
            org_inv_dic = pickle.load(f)

    # Calculate the distance matrix
    if not os.path.isfile(os.path.join(FLAGS.data_dir, 'aux_files', 'small_dist_counter_%s_%d.npy' % (FLAGS.data, MAX_VOCAB_SIZE))):
        print('small dist counter not exists, create and save...')
        dist_mat = utils.compute_dist_matrix(org_dic, FLAGS.data, MAX_VOCAB_SIZE, data_dir=FLAGS.data_dir)
        print('dist matrix created!')
        small_dist_mat = utils.create_small_embedding_matrix(dist_mat, MAX_VOCAB_SIZE, threshold=1.5, retain_num=50)
        print('small dist counter created!')
        np.save(os.path.join(FLAGS.data_dir, 'aux_files', 'small_dist_counter_%s_%d.npy' % (FLAGS.data, MAX_VOCAB_SIZE)), small_dist_mat)
    else:
        print('small dist counter exists, loading...')
        small_dist_mat = np.load(os.path.join(FLAGS.data_dir, 'aux_files', 'small_dist_counter_%s_%d.npy' % (FLAGS.data, MAX_VOCAB_SIZE)))

    if not os.path.isfile(os.path.join(FLAGS.data_dir, 'aux_files', 'embeddings_glove_%s_%d.npy' % (FLAGS.data, MAX_VOCAB_SIZE))):
        print('embeddings glove not exists, creating...')
        glove_model = utils.loadGloveModel('glove.840B.300d.txt', data_dir=FLAGS.data_dir)
        glove_embeddings, _ = utils.create_embeddings_matrix(glove_model, org_dic, dataset=FLAGS.data, data_dir=FLAGS.data_dir)
        print("embeddings glove created!")
        np.save(os.path.join(FLAGS.data_dir, 'aux_files', 'embeddings_glove_%s_%d.npy' % (FLAGS.data, MAX_VOCAB_SIZE)), glove_embeddings)
    else:
        print('embeddings glove exists, loading...')
        glove_embeddings = np.load(os.path.join(FLAGS.data_dir, 'aux_files', 'embeddings_glove_%s_%d.npy' % (FLAGS.data, MAX_VOCAB_SIZE)))

    print("Over!!!")

if __name__ == '__main__':
    tf.app.run()


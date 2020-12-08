import numpy as np
import pickle
import tensorflow as tf
import csv
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os


def read_text(path, data_dir="./"):
    print("reading path: %s" % (data_dir + path))
    label_list = []
    clean_text_list = []
    if (
        path.startswith("ag_news")
        or path.startswith("dbpedia")
        or path.startswith("yahoo_answers")
    ):
        with open(data_dir + "%s.csv" % path, "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            count = 0
            for row in csv_reader:
                count += 1
                label_list.append(int(row[0]) - 1)
                text = " . ".join(row[1:]).lower()
                clean_text_list.append(" ".join(text_to_tokens(text)))
    else:
        raise NotImplementedError
    return clean_text_list, label_list


def build_dict(dataset, vocab_size=50000, data_dir="./"):
    """
    The most frequently occurring words in the data set constitute the dictionary.
    Words that do not appear in the dictionary are all mapped to `UNK` with word id 0.
    """
    tokenizer = Tokenizer()
    train_text, _ = read_text(dataset + "/train", data_dir=data_dir)
    tokenizer.fit_on_texts(train_text)
    dic = dict()
    dic["UNK"] = 0
    inv_dict = dict()
    inv_dict[0] = "UNK"
    for word, idx in tokenizer.word_index.items():
        if idx <= vocab_size:
            inv_dict[idx] = word
            dic[word] = idx
    return dic, inv_dict, tokenizer


def load_dictionary(dataset, max_vocab_size, data_dir="./"):
    with open(
        (data_dir + "aux_files/org_dic_%s_%d.pkl" % (dataset, max_vocab_size)), "rb"
    ) as f:
        org_dic = pickle.load(f)
    with open(
        (data_dir + "aux_files/org_inv_dic_%s_%d.pkl" % (dataset, max_vocab_size)), "rb"
    ) as f:
        org_inv_dic = pickle.load(f)
    return org_dic, org_inv_dic


def loadGloveModel(gloveFile, data_dir="./"):
    """
    Load the glove model / glove model after counter-fitting.
    """
    print("Loading Glove Model")
    f = open(os.path.join(data_dir, gloveFile), "r", encoding="utf-8")
    model = {}
    for line in f:
        row = line.strip().split(" ")
        word = row[0]
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def create_embeddings_matrix(
    glove_model,
    dictionary,
    full_dictionary=None,
    embedding_size=300,
    dataset=None,
    data_dir="./",
):
    embedding_matrix = np.zeros(shape=((embedding_size, len(dictionary))))
    cnt = 0
    unfound_ids = []
    unfound_words = []
    for w, i in dictionary.items():
        if not w in glove_model:
            cnt += 1
            unfound_ids.append(i)
            unfound_words.append(w)
        else:
            embedding_matrix[:, i] = glove_model[w]
    print("Number of not found words = ", cnt)
    if cnt != 0 and dataset is not None:
        f = open(
            os.path.join(data_dir, "aux_files", "unfound_words_%s.txt" % (dataset)),
            "w",
            encoding="utf-8",
        )
        f.write(" ".join(unfound_words))
        f.close()
    return embedding_matrix, unfound_ids


def load_embeddings_matrix(dataset, max_vocab_size, data_dir="./"):
    glove_embeddings = np.load(
        data_dir + "aux_files/embeddings_glove_%s_%d.npy" % (dataset, max_vocab_size)
    )
    return glove_embeddings


def compute_dist_matrix(dic, dataset, vocab_size=50000, data_dir="./"):
    """
    Create a distance matrix of size (vacab_size+1, vocab_size+1),
    and record the distance between two words in the GloVe embedding space after counter-fitting.
    The distances related to `UNK` (word id=0) are set to INFINITY.
    """
    INFINITY = 100000
    embedding_matrix, missed = None, None
    if not os.path.isfile(
        os.path.join(
            data_dir,
            "aux_files",
            "embeddings_counter_%s_%d.npy" % (dataset, vocab_size),
        )
    ):
        print("embeddings_counter_%s_%d.npy" % (dataset, vocab_size) + " not exists.")
        glove_tmp = loadGloveModel("counter-fitted-vectors.txt", data_dir=data_dir)
        embedding_matrix, missed = create_embeddings_matrix(
            glove_tmp, dic, data_dir=data_dir
        )
        np.save(
            os.path.join(
                data_dir,
                "aux_files",
                "embeddings_counter_%s_%d.npy" % (dataset, vocab_size),
            ),
            embedding_matrix,
        )
        np.save(
            os.path.join(
                data_dir,
                "aux_files",
                "missed_embeddings_counter_%s_%d.npy" % (dataset, vocab_size),
            ),
            missed,
        )
    else:
        embedding_matrix = np.load(
            os.path.join(
                data_dir,
                "aux_files",
                "embeddings_counter_%s_%d.npy" % (dataset, vocab_size),
            )
        )
        missed = np.load(
            os.path.join(
                data_dir,
                "aux_files",
                "missed_embeddings_counter_%s_%d.npy" % (dataset, vocab_size),
            )
        )

    embedding_matrix = embedding_matrix.astype(np.float32)
    c_ = -2 * np.dot(embedding_matrix.T, embedding_matrix)
    a = np.sum(np.square(embedding_matrix), axis=0).reshape((1, -1))
    b = a.T
    dist = a + b + c_
    dist[0, :] = INFINITY
    dist[:, 0] = INFINITY
    dist[missed, :] = INFINITY
    dist[:, missed] = INFINITY
    print("success to compute distance matrix!")
    return dist


def create_small_embedding_matrix(
    dist_mat, MAX_VOCAB_SIZE, threshold=1.5, retain_num=50
):
    """
    Create the synonym matrix. 
    The i-th row represents the synonyms of the word with id i and their distances.
    """
    small_embedding_matrix = np.zeros(shape=((MAX_VOCAB_SIZE + 1, retain_num, 2)))
    for i in range(MAX_VOCAB_SIZE + 1):
        if i % 1000 == 0:
            print("%d/%d processed." % (i, MAX_VOCAB_SIZE))
        dist_order = np.argsort(dist_mat[i, :])[1 : 1 + retain_num]
        dist_list = dist_mat[i][dist_order]
        mask = np.ones_like(dist_list)
        if threshold is not None:
            mask = np.where(dist_list < threshold)
            dist_order, dist_list = dist_order[mask], dist_list[mask]
        n_return = len(dist_order)
        dist_order_arr = np.pad(
            dist_order, (0, retain_num - n_return), "constant", constant_values=(-1, -1)
        )
        dist_list_arr = np.pad(
            dist_list, (0, retain_num - n_return), "constant", constant_values=(-1, -1)
        )
        small_embedding_matrix[i, :, 0] = dist_order_arr
        small_embedding_matrix[i, :, 1] = dist_list_arr
    return small_embedding_matrix


def load_dist_mat(dataset, max_vocab_size, data_dir="./"):
    dist_mat = np.load(
        (
            data_dir
            + "aux_files/small_dist_counter_%s_%d.npy" % (dataset, max_vocab_size)
        )
    )
    return dist_mat


def text_to_tokens(text):
    """
    Clean the raw text.
    """
    spliter = re.split(
        r"([\'\#\ \!\"\$\%\&\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\t\n])",
        text,
    )
    tokens = [token for token in filter(lambda x: (x != "" and x != " "), spliter)]
    return tokens


def text_encoder(texts, org_dic, maxlen):
    """
    Map the raw text to word id sequence.
    """
    seqs = []
    seqs_mask = []
    for text in texts:
        words = text.split(" ")
        mask = []
        for i in range(len(words)):
            words[i] = org_dic[words[i]] if words[i] in org_dic else 0
            mask.append(1)
        seqs.append(words)
        seqs_mask.append(mask)
    seqs = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")
    seqs_mask = pad_sequences(
        seqs_mask, maxlen=maxlen, padding="post", truncating="post", value=0
    )
    return seqs, seqs_mask


def calculate_diff(s1, s2):
    count = 0
    s1_split = s1.split()
    s2_split = s2.split()
    if len(s1_split) != len(s2_split):
        print("Length mismatch\n" + s1 + "\n" + s2)
    else:
        for j in range(len(s1_split)):
            if s1_split[j] != s2_split[j]:
                count += 1
    return count


def calculate_diff_for_array(a, b):
    return np.sum(np.not_equal(np.array(a), np.array(b)))


def generate_model_save_path(timestamp, dataset, train_type):
    return "%s_%s_%s" % (timestamp, dataset, train_type)

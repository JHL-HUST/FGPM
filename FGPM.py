import numpy as np
import tensorflow as tf
from text_cnn import textcnn, compute_loss
from text_rnn import textrnn
from text_birnn import textbirnn


def find_synonym(xs, dist_mat, batch_size, word_max_len, threshold=0.5):
    xs = tf.expand_dims(xs, -1)
    synonyms = tf.gather_nd(dist_mat[:, :, 0], xs)
    synonyms_dist = tf.gather_nd(dist_mat[:, :, 1], xs)
    synonyms = tf.where(synonyms_dist <= threshold, synonyms, tf.zeros_like(synonyms))
    synonyms = tf.where(synonyms >= 0, synonyms, tf.zeros_like(synonyms))
    return synonyms


def FGPM(
    xs,
    ys,
    xs_mask,
    dataset,
    model,
    max_iter,
    num_classes,
    dist_mat,
    grad_update_interval,
    dis_threshold=0.5,
    sn=4,
    max_perturbed_percent=0.25,
    embedding_size=300,
    xs_org=None,
):

    adv_xs = xs  # batch_size, word_max_len, embedding_size
    if xs_org is None:
        xs_org = xs
    batch_size, word_max_len = tf.unstack(tf.shape(xs))
    modified_mask = tf.zeros_like(xs_mask)
    words_num = tf.reduce_sum(xs_mask, axis=-1)
    synonyms = tf.cast(
        find_synonym(
            xs_org, dist_mat, batch_size, word_max_len, threshold=dis_threshold
        ),
        tf.int32,
    )

    query = eval(model)

    def stop(adv_xs, modified_mask, ys, synonyms, i):
        return tf.less(i, max_iter)
    
    def one_step_attack(adv_xs, modified_mask, ys, synonyms, i):
        embeddings, embedded_chars, predictions, logits = query(
            adv_xs, 1.0, dataset, reuse=True
        )

        loss = compute_loss(logits, ys, num_classes)

        modified_num = tf.reduce_sum(modified_mask, axis=-1)
        modified_ratio = tf.divide(modified_num + 1, words_num)

        # The samples that have been misclassified or whose perturbation have exceeded the maximum threshold will no longer conduct synonym substituion.
        unsuccessful_mask = tf.logical_and(
            tf.equal(predictions, ys),
            tf.less_equal(modified_ratio, max_perturbed_percent),
        )

        # Step 1: Get gradient matrix.
        Jacobian = tf.gradients(loss, embedded_chars)[0]

        # Step 2: Compute Projection.
        synonyms_embed = tf.gather_nd(embeddings, tf.expand_dims(synonyms, -1))
        xs_embed = tf.expand_dims(embedded_chars, -2)
        Jacobian = tf.expand_dims(Jacobian, -2)
        projection = tf.reduce_sum(
            tf.multiply(synonyms_embed - xs_embed, Jacobian), axis=-1
        )

        # Step 3: Mask Projection. Substitution can only occur on known words.
        synonym_mask = tf.cast(tf.greater_equal(0, synonyms), tf.float32)
        inf = tf.fill([batch_size, word_max_len, sn], -1000000.0)
        delta_dense = synonym_mask * inf
        projection = tf.add(projection, delta_dense)

        # Step 4: Subsitution.
        _, pos = tf.nn.top_k(
            tf.reduce_max(projection, axis=-1), k=grad_update_interval
        )
        serial = tf.tile(
            tf.expand_dims(tf.range(0, batch_size, 1), -1), [1, grad_update_interval]
        )
        indices = tf.stack([serial, pos], axis=-1)
        indices_m = tf.boolean_mask(indices, unsuccessful_mask, axis=0)
        indices_m = tf.reshape(tf.cast(indices_m, tf.int64), [-1, 2])
        origin = tf.gather_nd(adv_xs, indices_m)
        synonym_pos = tf.expand_dims(
            tf.gather_nd(tf.argmax(projection, axis=2), indices_m), -1
        )
        synonym_indices_m = tf.concat([indices_m, synonym_pos], -1)
        synonym = tf.gather_nd(synonyms, synonym_indices_m)
        delta = tf.SparseTensor(indices_m, synonym - origin, [batch_size, word_max_len])
        adv_xs = adv_xs + tf.sparse_tensor_to_dense(delta, validate_indices=False)

        # Step 5: Record perturbed positions and mask used synonyms.
        updates = tf.fill(tf.expand_dims(tf.shape(indices_m)[0], axis=-1), 1)
        # modified_mask = tf.tensor_scatter_nd_update(modified_mask, indices_m, updates)
        delta = tf.SparseTensor(
            indices_m, updates - tf.gather_nd(modified_mask, indices_m), [batch_size, word_max_len] 
        )
        modified_mask = modified_mask + tf.sparse_tensor_to_dense(delta, validate_indices=False)
        delta = tf.SparseTensor(
            synonym_indices_m, -synonym, [batch_size, word_max_len, sn]
        )
        synonyms = synonyms + tf.sparse_tensor_to_dense(delta, validate_indices=False)

        i = tf.add(i, 1)

        return adv_xs, modified_mask, ys, synonyms, i

    i = tf.constant(0)

    adv_xs, _, _, _, _ = tf.while_loop(stop, one_step_attack, [adv_xs, modified_mask, ys, synonyms, i])

    _, _, predictions, _ = query(adv_xs, 1.0, dataset, reuse=True)
    suc_index = tf.not_equal(predictions, ys)
    
    return adv_xs, suc_index, predictions

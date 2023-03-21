"""Automatic label search helpers."""

import itertools
import torch
import tqdm
import multiprocessing
import numpy as np
import scipy.spatial as spatial
import scipy.special as special
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)

da_num = 3

def select_likely_words(train_logits, train_labels, k_likely=1000, vocab=None, is_regression=False):
    """Pre-select likely words based on conditional likelihood."""
    indices = []
    if is_regression:
        median = np.median(train_labels)
        train_labels = (train_labels > median).astype(np.int)
    num_labels = np.max(train_labels) + 1
    for idx in range(num_labels):
        label_logits = train_logits[train_labels == idx]
        scores = label_logits.mean(axis=0)
        kept = []
        for i in np.argsort(-scores):
            text = vocab[i]
            if not text.startswith("Ġ"):
                continue
            kept.append(i)
        indices.append(kept[:k_likely])
    return indices


def select_neighbors(distances, k_neighbors, valid):
    """Select k nearest neighbors based on distance (filtered to be within the 'valid' set)."""
    indices = np.argsort(distances)
    neighbors = []
    for i in indices:
        if i not in valid:
            continue
        neighbors.append(i)
    if k_neighbors > 0:
        return neighbors[:k_neighbors]
    return neighbors


def init(train_logits, train_labels):
    global logits, labels
    logits = train_logits
    labels = train_labels


def eval_pairing_acc(pairing):
    global logits, labels
    # print("logits :", logits)
    # print("labels :", labels)

    label_logits = np.take(logits, pairing, axis=-1)

    # print("label_logits :", label_logits)

    preds = np.argmax(label_logits, axis=-1)

    # print("preds :", preds)

    correct = np.sum(preds == labels)

    # print("correct :", correct)

    return correct / len(labels)

def eval_mul2one_pairing_acc(pairing):
    global logits, labels
    # label_logits = np.take(logits, pairing, axis=-1)
    # preds = np.argmax(label_logits, axis=-1)
    # correct = np.sum(preds == labels)
    # return correct / len(labels)

    # print("logits :", logits)
    # print("labels :", labels)

    index_list = []
    for index_tuple in pairing:
        index_list += list(index_tuple)
    # print("pairing :", pairing)
    # print("index_list :", index_list)

    label_logits = np.take(logits, index_list, axis=-1)
    # print("label_logits :", label_logits)
    preds = np.argmax(label_logits, axis=-1)
    # print("preds :", preds)
    preds = np.floor(preds/3)
    # print("preds :", preds)
    correct = np.sum(preds == labels)
    # print("correct :", correct)
    return correct / len(labels)



def eval_pairing_corr(pairing):
    global logits, labels
    if pairing[0] == pairing[1]:
        return -1
    label_logits = np.take(logits, pairing, axis=-1)
    label_probs = special.softmax(label_logits, axis=-1)[:, 1]
    pearson_corr = stats.pearsonr(label_probs, labels)[0]
    return pearson_corr


def find_labels(
    model,
    train_logits,
    train_labels,
    seed_labels=None,
    k_likely=1000,
    k_neighbors=None,
    top_n=-1,
    vocab=None,
    is_regression=False,
):
    # Get top indices based on conditional likelihood using the LM.
    likely_indices = select_likely_words(
        train_logits=train_logits,
        train_labels=train_labels,
        k_likely=k_likely,
        vocab=vocab,
        is_regression=is_regression)

    # print("2")

    logger.info("Top labels (conditional) per class:")
    for i, inds in enumerate(likely_indices):
        logger.info("\t| Label %d: %s", i, ", ".join([vocab[i] for i in inds[:10]]))
        print("\t| Label %d: %s", i, ", ".join([vocab[i] for i in inds]))
        # print("0")

    # print("1")

    # Convert to sets.
    valid_indices = [set(inds) for inds in likely_indices]

    # If specified, further re-rank according to nearest neighbors of seed labels.
    # Otherwise, keep ranking as is (based on conditional likelihood only).
    if seed_labels:
        assert(vocab is not None)
        seed_ids = [vocab.index(l) for l in seed_labels]
        vocab_vecs = model.lm_head.decoder.weight.detach().cpu().numpy()
        seed_vecs = np.take(vocab_vecs, seed_ids, axis=0)

        # [num_labels, vocab_size]
        label_distances = spatial.distance.cdist(seed_vecs, vocab_vecs, metric="cosine")

        # Establish label candidates (as k nearest neighbors).
        label_candidates = []
        logger.info("Re-ranked by nearest neighbors:")
        for i, distances in enumerate(label_distances):
            label_candidates.append(select_neighbors(distances, k_neighbors, valid_indices[i]))
            logger.info("\t| Label: %s", seed_labels[i])
            logger.info("\t| Neighbors: %s", " ".join([vocab[idx] for idx in label_candidates[i]]))
    else:
        label_candidates = likely_indices
        print("label_candidates : ", label_candidates)


    # Brute-force search all valid pairings.
    pairings = list(itertools.product(*label_candidates))

    '''
    # have intersection
    multi_candidates = []
    for l in label_candidates:
        multi_candidates.append(list(itertools.combinations(l, 3)))
    '''
    
    # '''
    # no intersection
    label_candidates_set_list = []
    for label_candidate_list in label_candidates:
        label_candidates_set_list.append(set(label_candidate_list))
    
    label_candidates_set_list_no_intersection = []
    for i in range(len(label_candidates_set_list)):
        tmp = label_candidates_set_list[i]
        for j in range(len(label_candidates_set_list) - 1):
            k = (i + j + 1) % len(label_candidates_set_list)
            tmp -= label_candidates_set_list[k]
        label_candidates_set_list_no_intersection.append(tmp)

    label_candidates_list_no_intersection = []
    for i in label_candidates_set_list_no_intersection:
        label_candidates_list_no_intersection.append(list(i))

    print("label_candidates_set_list_no_intersection :", label_candidates_set_list_no_intersection)
    print("label_candidates_list_no_intersection :", label_candidates_list_no_intersection)

    multi_candidates = []
    for l in label_candidates_list_no_intersection:
        multi_candidates.append(list(itertools.combinations(l, 3)))
    # '''

    # print("multi_candidates : ", multi_candidates)

    mul2one_pairings = list(itertools.product(*multi_candidates))
    # print("pairings : ", pairings)
    # print("mul2one_pairings : ", mul2one_pairings)


    # if is_regression:
    #     eval_pairing = eval_pairing_corr
    #     metric = "corr"
    # else:
    #     eval_pairing = eval_pairing_acc
    #     metric = "acc"

    # # Score each pairing.
    # pairing_scores = []
    # with multiprocessing.Pool(initializer=init, initargs=(train_logits, train_labels)) as workers:
    #     with tqdm.tqdm(total=len(pairings)) as pbar:
    #         chunksize = max(10, int(len(pairings) / 1000))
    #         for score in workers.imap(eval_pairing, pairings, chunksize=chunksize):
    #             pairing_scores.append(score)
    #             pbar.update()

    # # Take top-n.
    # best_idx = np.argsort(-np.array(pairing_scores))[:top_n]
    # best_scores = [pairing_scores[i] for i in best_idx]
    # best_pairings = [pairings[i] for i in best_idx]

    # logger.info("Automatically searched pairings:")
    # for i, indices in enumerate(best_pairings):
    #     logger.info("\t| %s (%s = %2.2f)", " ".join([vocab[j] for j in indices]), metric, best_scores[i])
    #     print("\t| %s (%s = %2.2f)", " ".join([vocab[j] for j in indices]), metric, best_scores[i])

    # """
    eval_mul2one_pairing = eval_mul2one_pairing_acc
    mul2one_metric = "acc"

    pairing_scores = []
    with multiprocessing.Pool(initializer=init, initargs=(train_logits, train_labels)) as workers:
        with tqdm.tqdm(total=len(mul2one_pairings)) as pbar:
            chunksize = max(10, int(len(mul2one_pairings) / 1000))
            for score in workers.imap(eval_mul2one_pairing, mul2one_pairings, chunksize=chunksize):
                pairing_scores.append(score)
                pbar.update()
    # """

    # best_idx = np.argsort(-np.array(pairing_scores))[:top_n]
    # best_idx = np.argsort(-np.array(pairing_scores))[:200]
    best_idx = np.argsort(-np.array(pairing_scores))[:top_n]
    best_scores = [pairing_scores[i] for i in best_idx]
    best_pairings = [mul2one_pairings[i] for i in best_idx]

    best_pairings_vocab = []
    for index_tuple_pair in best_pairings:
        vocab_pair = []
        for index_tuple in index_tuple_pair:
            vocab_pair.append([vocab[j] for j in index_tuple])
        best_pairings_vocab.append(vocab_pair)

    # print("best_pairings_vocab :", best_pairings_vocab)

    logger.info("Automatically searched pairings:")
    for i, indices in enumerate(best_pairings):
        # logger.info("\t| %s (%s = %2.2f)", " ".join([vocab[j] for j in indices]), mul2one_metric, best_scores[i])
        print(best_pairings_vocab[i], mul2one_metric, best_scores[i])

    return best_pairings

#!/usr/bin/python
"""
Metrics for evaluating hierachical multilabel classification performance.

Ancestor/descendant metrics inspired by Sokolova, M., & Lapalme, G. (2009).
A systematic analysis of performance measures for classification tasks. 
Information Processing & Management, 45(4), 427-437. 
doi:10.1016/j.ipm.2009.03.002
"""

from __future__ import division, print_function

import dill
import numpy as np
from sklearn import metrics as skmetrics
from sklearn.preprocessing import MultiLabelBinarizer

from average_precision import apk, mapk

EXTEND_PRED = True


def calculate_mapk_extended_true_pred(class_hierarchy, y_true, y_pred, k):
    apks = []

    for true, pred in zip(y_true, y_pred):
        true_set = set(list(true) + class_hierarchy._get_ancestors_of(true))
        pred_set = set(list(pred) + class_hierarchy._get_ancestors_of(pred))

        _apk = apk(true_set, pred_set, k)
        apks.append(_apk)
    return np.mean(apks)


def calculate_mapk_extended_true(class_hierarchy, y_true, y_pred, k):
    apks = []

    for true, pred in zip(y_true, y_pred):
        true_set = set(list(true) + class_hierarchy._get_ancestors_of(true))
        pred_set = set(list(pred))

        _apk = apk(true_set, pred_set, k)
        apks.append(_apk)

    return np.mean(apks)


def calculate_mapk(class_hierarchy, y_true, y_pred, k):
    _mapk = mapk(y_true, y_pred, k)

    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        lambda x: list(set(x)), y_true, y_pred
    )

    return _mapk


# General Scores
# Average accuracy
def accuracy_score(class_hierarchy, y_true, y_pred):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        lambda x: list(set(x)), y_true, y_pred
    )

    if predicted_sum > 0:
        return intersection_sum / predicted_sum
    else:
        return 0


def sk_accuracy_score(class_hierarchy, y_true, y_pred):
    multibinarizer = MultiLabelBinarizer()

    _y_true = multibinarizer.fit(y_true).transform(y_true)
    _y_pred = multibinarizer.transform(y_pred)

    return skmetrics.accuracy_score(_y_true, _y_pred)


def accuracy_scores(class_hierarchy, y_true, y_pred):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        lambda x: list(set(x)), y_true, y_pred
    )

    return 1 + intersection_sum / 1 + predicted_sum


# Hierarchy Precision / Recall
def _aggregate_class_sets(set_function, y_true, y_pred, return_sums=False):
    intersection_sums = []
    predicted_sums = []
    true_sums = []
    intersection_sum = 0
    true_sum = 0
    predicted_sum = 0
    cnt = 0

    try:
        y_pred = y_pred.to_numpy().ravel()
    except Exception as ex:
        pass  # already raveled

    for true, pred in zip(y_true, y_pred):
        true_set = set(list(true) + set_function(true))
        if EXTEND_PRED:
            pred_set = set(list(pred) + set_function(pred))
        else:
            pred_set = set(list(pred) + pred)
        if len(pred) < len(pred_set):
            cnt += 1
        intersection_sum += len(true_set.intersection(pred_set))
        true_sum += len(true_set)
        predicted_sum += len(pred_set)

        intersection_sums.append(len(true_set.intersection(pred_set)))
        true_sums.append(len(true_set))
        predicted_sums.append(len(pred_set))

    if return_sums:
        return (
            true_sum,
            predicted_sum,
            intersection_sum,
            intersection_sums,
            true_sums,
            predicted_sums,
        )

    return (true_sum, predicted_sum, intersection_sum)


def save_sums(class_hierarchy, y_true, y_pred, FEATURES):
    (
        true_sum,
        predicted_sum,
        intersection_sum,
        intersection_sums,
        true_sums,
        predicted_sums,
    ) = _aggregate_class_sets(class_hierarchy._get_ancestors_of, y_true, y_pred, True)

    sums_path = f"intersection_sums_{FEATURES}_ext{EXTEND_PRED}.pickle"
    with open(sums_path, "wb") as f:
        dill.dump((intersection_sums, true_sums, predicted_sums), f)
    print(f"Dumped {sums_path}")


# Ancestors Scores (Super Class)
# Precision
def precision_score_ancestors(class_hierarchy, y_true, y_pred):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_ancestors_of, y_true, y_pred
    )
    if predicted_sum > 0:
        return (1 + intersection_sum) / (1 + predicted_sum)
    else:
        return 0


# Recall
def recall_score_ancestors(class_hierarchy, y_true, y_pred):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_ancestors_of, y_true, y_pred
    )
    if true_sum > 0:
        return (1 + intersection_sum) / (1 + true_sum)
    else:
        return 0


# Descendants Scores (Sub Class)
# Precision
def precision_score_descendants(class_hierarchy, y_true, y_pred):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_descendants_of, y_true, y_pred
    )
    return (1 + intersection_sum) / (1 + predicted_sum)


# Recall
def recall_score_descendants(class_hierarchy, y_true, y_pred):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_descendants_of, y_true, y_pred
    )
    return (1 + intersection_sum) / (1 + true_sum)


# Branch Scores (Sub and Super Classes)
# Precision
def precision_score_branch(class_hierarchy, y_true, y_pred):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_descendants_ascendants_of, y_true, y_pred
    )
    return (1 + intersection_sum) / (1 + predicted_sum)


# Recall
def recall_score_branch(class_hierarchy, y_true, y_pred):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_descendants_ascendants_of, y_true, y_pred
    )
    return (1 + intersection_sum) / (1 + true_sum)


# Hierarchy Fscore
def _fbeta_score_class_sets(set_function, y_true, y_pred, beta=1):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        set_function, y_true, y_pred
    )
    precision = (1 + intersection_sum) / (1 + predicted_sum)
    recall = (1 + intersection_sum) / (1 + true_sum)
    return ((beta ** 2 + 1) * precision * recall) / ((beta ** 2 * precision) + recall)


def f1_score_ancestors(class_hierarchy, y_true, y_pred):
    return _fbeta_score_class_sets(class_hierarchy._get_ancestors_of, y_true, y_pred)


def f1_score_descendants(class_hierarchy, y_true, y_pred):
    return _fbeta_score_class_sets(class_hierarchy._get_descendants_of, y_true, y_pred)


def f1_score_branch(class_hierarchy, y_true, y_pred):
    return _fbeta_score_class_sets(
        class_hierarchy._get_descendants_ascendants_of, y_true, y_pred
    )


def classification_report(class_hierarchy, y_true, y_pred):
    precision_anc = precision_score_ancestors(class_hierarchy, y_true, y_pred)
    recall_anc = recall_score_ancestors(class_hierarchy, y_true, y_pred)
    f1_score_anc = f1_score_ancestors(class_hierarchy, y_true, y_pred)
    precision_desc = precision_score_descendants(class_hierarchy, y_true, y_pred)
    recall_desc = recall_score_descendants(class_hierarchy, y_true, y_pred)
    f1_score_desc = f1_score_descendants(class_hierarchy, y_true, y_pred)
    precision_branch = precision_score_branch(class_hierarchy, y_true, y_pred)
    recall_branch = recall_score_branch(class_hierarchy, y_true, y_pred)
    _f1_score_branch = f1_score_branch(class_hierarchy, y_true, y_pred)

    accuracy = accuracy_score(class_hierarchy, y_true, y_pred)

    print("=PRECISION" + "=" * 47)
    print(f"precision_anc : {precision_anc}")
    print(f"precision_desc: {precision_desc}")
    print(f"precision_brnc: {precision_branch}")
    print("=RECALL" + "=" * 50)
    print(f"recall_anc : {recall_anc}")
    print(f"recall_desc: {recall_desc}")
    print(f"recall_brnc: {recall_branch}")
    print("=F1" + "=" * 54)
    print(f"f1_score_anc : {f1_score_anc}")
    print(f"f1_score_desc: {f1_score_desc}")
    print(f"f1_score_brnc: {_f1_score_branch}")
    print("=EXACT MATCH" + "=" * 45)
    print(f"exact match accuracy: {accuracy}")


def classification_report_topk(class_hierarchy, y_true, y_pred, k):
    extended_mapk = calculate_mapk_extended_true_pred(
        class_hierarchy, y_true, y_pred, k
    )
    extended_mapk_true = calculate_mapk_extended_true(
        class_hierarchy, y_true, y_pred, k
    )
    mapk = calculate_mapk(class_hierarchy, y_true, y_pred, k)

    print(f"K =: {k}")
    print(f"MAP@K : {mapk}")
    print(f"MAP@K extended : {extended_mapk}")
    print(f"MAP@K extended_true: {extended_mapk_true}")

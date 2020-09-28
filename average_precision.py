import numpy as np


def get_true_pred_topk(labels, y_pred_prob, y_true, k=5):
    """Gets top K predictions from the given list
    of probabilities

    Parameters
    ----------
    labels : list(str)
        A list of all of the labels, sorted ascending

    y_pred_prob : list(float)
        A list of all of the labels probabilities

    y_true : list(int)
        A list of all of the true, labels in a one-hot
        encoded format

    k: int
        The number of top predictions to extract

    Returns
    -------
    tuple(list(str), list(str))
        Lists of true and predicted labels for each example
    """
    # map one-hot encoding to index list
    def onehot_to_ind(l):
        return [index for index, value in enumerate(l) if value == 1]

    indices = list(
        map(
            lambda x: sorted(range(len(x)), key=lambda k: x[k], reverse=True)[:k],
            y_pred_prob,
        )
    )
    indices_true = list(map(onehot_to_ind, y_true))

    labels_pred = list(map(lambda x: [labels[y] for y in x], indices))
    labels_true = list(map(lambda x: [labels[y] for y in x], indices_true))

    print(f"true: {labels_true[0]}")
    print(f"pred: {labels_pred[0]}")

    return labels_true, labels_pred


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The number of top predictions to extract

    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not isinstance(predicted, list):
        predicted = list(predicted)
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The number of top predictions to extract
        
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

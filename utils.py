# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import hmc
from hina_loader import Article


def flatten_dataset(dataset, depth):
    """Flattens the provided dataset up to a certain depth.
    The process will associate leaf nodes to their parents
    up until the specified depth.

    Parameters
    ----------
    dataset : list(namedtuple(Article))
        The list of parsed articles, each consisting of
        the article text, and the topic and genre labels.

    depth : int
        The desired depth up until which the topics should
        be flattened.

    Returns
    -------
    new_dataset : list(namedtuple(Article))
        The list of articles with topics flattened up to the
        `depth` level.
    """
    new_dataset = []
    for article in dataset:
        topics = set()
        for theme in article.topics:
            groups = theme.split("|")
            topics.add("|".join(groups[0:depth]))
        new_article = Article(text=article.text, topics=topics, genre=article.genre)
        new_dataset.append(new_article)
    return new_dataset


def flatten_remove_first_lvl(dataset, depth):
    """Flattens the provided dataset up to a certain depth.
    The process will associate leaf nodes to their parents
    up until the specified depth, and will not return the
    first-level topics (before the first `|` divider).

    Parameters
    ----------
    dataset : list(namedtuple(Article))
        The list of parsed articles, each consisting of
        the article text, and the topic and genre labels.

    depth : int
        The desired depth up until which the topics should
        be flattened.

    Returns
    -------
    new_dataset : list(namedtuple(Article))
        The list of articles with topics flattened up to the
        `depth` level.
    """
    new_dataset = []
    for article in dataset:
        topics = set()
        for theme in article.topics:
            groups = theme.split("|")
            if len(groups) < 2:
                break  # ignore first level articles
            topics.add("|".join(groups[0:depth]))
        new_article = Article(text=article.text, topics=topics, genre=article.genre)
        new_dataset.append(new_article)
    return new_dataset


def build_ch(dataset):
    """Builds the ClassHierarchy object from the given data
    provided as a list of Article namedtuple objects.

    Parameters
    ----------
    dataset : list(namedtuple(Article))
        The list of parsed articles, each consisting of
        the article text, and the topic and genre labels.

    Returns
    -------
    ch(ClassHierarchy)
        The ClassHierarchy objects, which contains the info about
        all of the labels, and their relationships.
    """
    topics_list = set()

    ch = hmc.ClassHierarchy("topics")
    for article in dataset:
        for theme in article.topics:
            groups = theme.split("|")
            # add first
            ch.add_node(groups[0].strip(), "topics")
            topics_list.add(groups[0].strip())
            for i, g in enumerate(groups[1:]):
                try:
                    ch.add_node(g.strip(), groups[i])
                    topics_list.add(g.strip())
                except Exception:
                    print(
                        f"FAILED ADD FOR topic: {g.strip()}"
                        f" into parent: {groups[i]}"
                    )
    return ch, topics_list


def build_ch(df):
    """Builds the ClassHierarchy object from the given data
    wrapped in a Pandas DataFrame.

    Parameters
    ----------
    df : DataFrame
        The Pandas dataframe which contains the preprocessed data

    Returns
    -------
    ch(ClassHierarchy)
        The ClassHierarchy objects, which contains the info about
        all of the labels, and their relationships.
    """
    topics_list = list(df)[5:]

    ch = hmc.ClassHierarchy("topics")
    for index in range(len(df)):  # for each article
        df_t = df.loc[[index]]
        criteria = (df_t == 1).any(axis=0)
        topics = list(df[criteria.index[criteria]])  # extract topics
        for topic in topics:
            groups = topic.split("|")
            # add root node
            parent = groups[0]
            ch.add_node(parent.strip(), "topics")
            for i, g in enumerate(groups[1:]):
                try:
                    # add topic with full name
                    topic_full = "|".join(groups[: (i + 2)])
                    parent = "|".join(groups[: (i + 1)])
                    ch.add_node(topic_full, parent.strip())
                except Exception:
                    print(
                        f"FAILED ADD FOR topic: {topic_full}" f" into parent: {parent}"
                    )

    return ch, topics_list


def prep_df_for_train(df, plot=False, topn=None):
    """Prepares the provided Pandas dataframe for training,
    stripping off articles with no labels and labels with low
    occurency. Can be set to return only the n-subset with
    the highest occurency rates.

    Parameters
    ----------
    df : DataFrame
        The Pandas dataframe which contains the preprocessed data

    plot : bool
        Whether the most frequent labels should be plotted

    topn : int
        If specified, will return only data labeled with topn
        most frequent labels

    ch_path : str
        Location of the class hierarchy file

    hc_path : str
        Location of the pretrained HierarchicalClassifier

    Returns
    -------
    tuple(df(DataFrame), y_n(DataFrame))
        DataFrames containing the input data and the accompanying
        labels, respectively
    """
    df = df.copy()

    topics = list(df.columns.values[5:])
    topics_df = pd.DataFrame(
        {"topics": list(topics), "count": list(df.iloc[:, 5:].sum().values)}
    )

    # selecting top n most frequent topics
    if topn is not None:
        df_freq = topics_df.nlargest(columns="count", n=topn)
    else:
        df_freq = topics_df

    top_n_cats = list(df_freq.iloc[:, 0])
    df_n = df[top_n_cats]

    # stripping off entries with no labels
    y_n = pd.DataFrame(df_n)
    print(f"df.shape: {df.shape}")
    print(f"y_n.shape: {y_n.shape}")
    y_n = y_n[~((y_n.iloc[:, 5:].sum(axis=1).values) == 0)]

    mask = df.index.isin(y_n.index.values)
    df = df.loc[mask]
    print(f"df.shape no labels: {df.shape}")
    print(f"y_n.shape no labels: {y_n.shape}")

    # stripping off labels with low occurency
    y_m = topics_df[(topics_df["count"] < 20)]

    mask = df.index.isin(y_m.index.values)
    df = df.loc[~mask]
    y_n = y_n[~mask]
    print(f"df.shape with low occurency mask: {df.shape}")
    print(f"y_n.shape with low occurency mask: {y_n.shape}")

    if plot:
        plt.figure(figsize=(15, 30))
        ax = sns.barplot(data=_df, x="count", y="topics", color="lightseagreen")
        ax.set(ylabel="Topic")
        plt.show()

    df.reset_index(inplace=True, drop=True)
    y_n.reset_index(inplace=True, drop=True)

    return df, y_n


def label_ids_to_labels(label_ids, labels):
    """Gets label strings from a list of theie ids

    Parameters
    ----------
    labels : list(int)
        A list of all of the specified label ids

    labels : list(str)
        A list of all of the labels, sorted ascending

    Returns
    -------
    list(str)
        List of the matching label strings
    """
    assert len(label_ids) == len(labels)
    out = []
    for index, id in enumerate(label_ids):
        if id:
            out.append(labels[index])
    return out


def get_true_pred(labels, y_pred_prob, y_true, threshold=0.5):
    """Gets label predictions from the given list
    of probabilities for the specified decision threshold

    Parameters
    ----------
    labels : list(str)
        A list of all of the labels, sorted ascending

    y_pred_prob : list(float)
        A list of all of the labels probabilities

    y_true : list(int)
        A list of all of the true, labels in a one-hot
        encoded format

    y_true : list(int)
        The desired desicion threshold

    Returns
    -------
    tuple(list(str), list(str))
        Lists of true and predicted labels for each example
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    def onehot_to_ind(l):
        return [index for index, value in enumerate(l) if value == 1]

    indices = list(map(onehot_to_ind, y_pred))
    indices_true = list(map(onehot_to_ind, y_true))

    labels_pred = list(map(lambda x: [labels[y] for y in x], indices))
    labels_true = list(map(lambda x: [labels[y] for y in x], indices_true))

    return labels_true, labels_pred

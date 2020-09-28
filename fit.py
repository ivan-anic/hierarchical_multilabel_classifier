import argparse
import pickle

import dill
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import hmc
from utils import label_ids_to_labels, prep_df_for_train


def fit(features, dataframe_path, ch_path, hmc_path):
    """Gets label predictions from the previously fit model

    Parameters
    ----------
    features : str
        The specified type of features - ["bert", "tfidf"]

    dataframe_path : str
        The path to the Pandas dataframe which contains the
        preprocessed data

    ch_path : str
        Location of the class hierarchy file

    hmc_path : str
        Path which points to the location where the fit
        HierarchicalMultilabelClassifier will be saved
    """
    df = pd.read_pickle(dataframe_path)
    topics = list(df)[5:]

    df, y = prep_df_for_train(df, None)
    print(y.shape)
    print(df.info())

    df.reset_index(inplace=True, drop=True)

    classifier = None
    # load features
    if features == "bert":
        classifier = "mlp"

        path = f"bert-multilingual/bert_train.npy"
        f = np.load(path, allow_pickle=True)
        print(
            f"Total: {len(f)}, tokens: {len(f[0]['embeddings'][0])}, embeds: ",
            f"{len(f[0]['embeddings'][1])}x{len(f[0]['embeddings'][1][0])}",
        )
        embeddings_list = f

        # parse from pickle
        embeddings = []
        tokens_list = []
        y = []

        for entry in embeddings_list:
            _y = label_ids_to_labels(entry["label_ids"], topics)

            y.append(_y)
            tokens, vectors = entry["embeddings"]

            tokens_list.append(tokens)
            embeddings.append(vectors)
            assert len(vectors) == 4
            assert len(vectors[0]) == 768

        assert len(embeddings) == len(y)
        print(f"Number of examples: {len(embeddings)}")

        f.close()

        X = embeddings
        print(len(X), len(X[0]), len(y))

        X = np.asarray(X).reshape(len(X), 3072)
        print(X.shape)

    if features == "tfidf":
        classifier = "lr"
        y = df["topics"]

        xtrain, xtest, ytrain, ytest = train_test_split(
            df["clean_text_tokenized"], y, test_size=0.2, random_state=42
        )
        xtrain, xdev, ytrain, ydev = train_test_split(
            xtrain, ytrain, test_size=0.25, random_state=42
        )
        print(f"train shape, {xtrain.shape}")
        print(f"dev shape, {xdev.shape}")
        print(f"test shape, {xtest.shape}")

        # create TF-IDF features
        tfidf_vectorizer = TfidfVectorizer(
            max_df=0.8, max_features=10000, preprocessor=" ".join
        )
        X = tfidf_vectorizer.fit_transform(xtrain)
        print(X.shape)

        X = pd.DataFrame(X.todense())
        X.reset_index(inplace=True, drop=True)
        y = ytrain
        y.reset_index(inplace=True, drop=True)

    # load class hierarchy
    ch = None
    with open(ch_path, "rb") as f:
        ch = pickle.load(f)

    clf = hmc.HierarchicalMultilabelClassifier(ch, classifier=classifier)
    clf = clf.fit(X, y)

    with open(hmc_path, "wb") as f:
        dill.dump(clf, f)

    print(
        f"Classifier ({classifier}) for {features} features "
        f"is fit and saved to {hmc_path}."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("features", choices=["bert", "tfidf"], help="Feature type")
    parser.add_argument(
        "-df",
        "--dataframe_path",
        help="Location of the Pandas DataFrame containing the preprocessed data",
        default="./dataset_2_with_topics_prot_4.pkl",
    )
    parser.add_argument(
        "-ch",
        "--class_hierarchy_path",
        help="Location of the class hierarchy file",
        default="./ch.pickle",
    )
    parser.add_argument(
        "-hmc",
        "--classifier_path",
        help="Location of the pretrained HierarchicalMultilabelClassifier",
        default="./trained/dth_trained_tfidf_lr.pickle",
    )
    args = parser.parse_args()

    fit(
        features=args.features,
        dataframe_path=args.dataframe_path or "./dataset_2_with_topics_prot_4.pkl",
        ch_path=args.class_hierarchy_path or "./ch.pickle",
        hmc_path=args.classifier_path or "./trained/hmc_trained_tfidf_lr.pickle",
    )


if __name__ == "__main__":
    main()

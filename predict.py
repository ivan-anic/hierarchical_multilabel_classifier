import argparse
import pickle

import dill
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import metrics
from utils import label_ids_to_labels, prep_df_for_trainnnn


def predict(features, stage, dataframe_path, ch_path, hmc_path):
    """Gets label predictions from the previously fit model

    Parameters
    ----------
    features : str
        The specified type of features - ["bert", "tfidf"]

    stage : str
        Predict stage; which data to use - ["dev", "test"]

    dataframe_path : str
        The path to the Pandas dataframe which contains the
        preprocessed data

    ch_path : str
        Location of the class hierarchy file

    hmc_path : str
        Location of the pretrained HierarchicalClassifier
    """
    df = pd.read_pickle(dataframe_path)
    topics = list(df)[5:]

    df, y = prep_df_for_trainnnn(df, None)
    print(y.shape)
    print(df.info())

    df.reset_index(inplace=True, drop=True)

    # load features
    if features == "bert":
        path = f"bert-multilingual/bert_{stage}.npy"
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

    elif features == "tfidf":
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
        xtrain = tfidf_vectorizer.fit_transform(xtrain)
        xdev = tfidf_vectorizer.transform(xdev)
        xtest = tfidf_vectorizer.transform(xtest)

        print(xdev.shape, xtest.shape)

        if stage == "dev":
            X = pd.DataFrame(xdev.todense())
            y = ydev
        elif stage == "test":
            X = pd.DataFrame(xtest.todense())
            y = ytest

        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)

    # load trained model
    clf = None
    if features == "tfidf":
        with open(hmc_path, "rb") as f:
            clf = dill.load(f)
    elif features == "bert":
        with open(hmc_path, "rb") as f:
            clf = dill.load(f)

    # load class hierarchy
    ch = None
    with open(ch_path, "rb") as f:
        ch = pickle.load(f)

    # get predictions
    ypred = clf.predict(X)

    metrics.classification_report(ch, y, pd.DataFrame(ypred))
    metrics.EXTEND_PRED = False
    print("=" * 100)
    metrics.classification_report(ch, y, pd.DataFrame(ypred))
    metrics.EXTEND_PRED = True

    ypred_topk = dth.predict_topk(Xdev, 5)
    metrics.classification_report_topk(ch, ydev, ypred_topk, 1)
    metrics.classification_report_topk(ch, ydev, ypred_topk, 3)
    metrics.classification_report_topk(ch, ydev, ypred_topk, 5)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "features", 
        choices=["bert", "tfidf"], 
        help="Feature type",
    )
    parser.add_argument(
        "stage",
        choices=["dev", "test"],
        help="Predict stage; which data to use",
    )
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

    predict(
        features=args.features,
        stage=args.stage,
        dataframe_path=args.dataframe_path,
        ch_path=args.class_hierarchy_path,
        hmc_path=args.classifier_path,
    )


if __name__ == "__main__":
    main()

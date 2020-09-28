import argparse
import json
import os
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from bert import modeling, tokenization
from bert_features import create_examples
from bert_model_utils import (
    file_based_convert_examples_to_features,
    file_based_input_fn_builder,
    model_fn_builder,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def create_input_fn_from_examples(
    examples,
    stage,
    base_working_path,
    max_seq_length,
    topics_length,
    force_build_tfrecord=False,
):
    """Gets label predictions from the previously fit model

    Parameters
    ----------
    examples : list(InputExample)
        The input exaples wrapped in the InputExample object

    stage : str
        Predict stage; which data to use - ["dev", "test"]

    base_working_path : str
        Root location of the directory which will contain the
        data in the .tfrecord file format

    max_seq_length : int
        The maximum sequence length for BERT

    topics_length : int
        The number of classes

    force_build_tfrecord : bool
        Whether the .tfrecord file should be (re)generated

    Returns
    -------
    input_fn : function
        An input function which will be passed to the estimator.
    """
    input_file = os.path.join(base_working_path, f"bert_{stage}.tf_record")
    if not os.path.exists(input_file) or force_build_tfrecord:
        open(input_file, "w").close()
        # generate train tf.record data if needed
        tfrecord_time_start = time()
        file_based_convert_examples_to_features(
            examples, max_seq_length, tokenizer, input_file
        )
        print(
            f"Finished generating {stage} set tf.record",
            f"in {time()-tfrecord_time_start} seconds.",
        )

    input_fn = file_based_input_fn_builder(
        input_file=input_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=False,
        len_labels=topics_length,
    )

    return input_fn


def generate_bert_embeddings(
    mode, input_fn, embeddings_list, label_ids_list, estimator, tokenizer, layer_indices
):
    """Gets label predictions from the previously fit model

    Parameters
    ----------
    mode : str
        The specified output embedding dimension -
        ["4x768", "512x768", "4x512x768"]

    input_fn : function
        The input function which feeds examples into the estimator

    embeddings_list : np.array
        Numpy array which will be populated with BERT text token
        representations

    label_ids_list : np.array
        Numpy array which will be populated with the output labels

    estimator : tf.estimator.Estimator
        The class which will generate the model predictions

    tokenizer : tokenization.FullTokenizer
        The BERT tokenizer

    layer_indices: list(int)
        Indices of BERT layers which should be taken into account
        when generating embeddings
        
    Returns
    -------
    tuple(np.array, np.array)
        Numpy arrays containing the word embeddings and
        accompanying class labels
    """
    cnt = 0
    for result in tqdm(estimator.predict(input_fn, yield_single_examples=True)):
        out = {}
        ids = result["input_ids"]
        label_ids = result["label_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)

        embeddings = []
        for (i, token) in enumerate(tokens):
            layers = []
            for (j, layer_index) in enumerate(layer_indices):
                layer_output = result["layer_output_%d" % j]
                layer_output_flat = np.array(
                    [x for x in layer_output[i : (i + 1)].flat]
                )
                layers.append(layer_output_flat)
            if mode == "4x768":
                embeddings.append(np.average(layers, axis=0))  # 4 x 768
            elif mode == "512x768":
                embeddings.append(sum(layers))  # 768 x 512
            elif mode == "4x512x768":
                embeddings.append((token, layers))  # 4 x 768 x 512

        embeddings_list[cnt] = embeddings
        label_ids_list[cnt] = label_ids

        if cnt < 1:
            if mode == "512x768":
                print(
                    f"token: {token}, shape[1]: ",  # 0,1 - first embed, 0 is token
                    f"{len(embeddings), len(embeddings[0])}, types: {type(layers[0][0])}",
                )
            elif mode == "4x512x768":
                print(
                    f"token: {token}, shape[1]: ",  # 0,1 - first embed, 0 is token
                    f"{len(embeddings), len(embeddings[0][1])}, types: {type(layers[0][0])}",
                )
            print(f"label_ids: {len(label_ids)}")
        cnt += 1

    return embeddings_list, label_ids_list


def generate_embeddings(args):
    """Generates a set of word embeddings from the final four BERT
    layers.

    Parameters
    ----------
    args : Namespace
        Parsed arguments from argparse, containing all of the input arguments
    """
    time_start = time()

    print(tf.__version__)
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )

    # data
    df = pd.read_pickle(args.dataframe_path)
    topics = list(df)[5:]

    xtrain, xtest, ytrain, ytest = train_test_split(
        df["clean_text"], df.iloc[:, 5:], test_size=0.2, random_state=42
    )
    xtrain, xdev, ytrain, ydev = train_test_split(
        xtrain, ytrain, test_size=0.25, random_state=42
    )
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    print(xtrain.shape, xdev.shape, ytrain.shape, ydev.shape)

    df_train = pd.concat([xtrain, ytrain], axis=1, ignore_index=True)
    df_dev = pd.concat([xdev, ydev], axis=1, ignore_index=True)
    df_test = pd.concat([xtest, ytest], axis=1, ignore_index=True)
    print(f"train shape: {df_train.shape}")
    print(f"val shape: {df_dev.shape}")
    print(f"test shape: {df_test.shape}")

    if args.stage == "train":
        examples = create_examples(df_train)
    if args.stage == "dev":
        examples = create_examples(df_dev)
    if args.stage == "test":
        examples = create_examples(df_test)

    input_fn = create_input_fn_from_examples(
        examples, args.stage, args.base_working_path, args.max_seq_length, len(topics)
    )

    # model init
    bert_vocab = args.base_path + "/bert_vocab.txt"
    bert_init_chckpnt = args.base_path + "/bert_model.ckpt"
    bert_config = args.base_path + "/config.json"

    tokenization.validate_case_matches_checkpoint(True, bert_init_chckpnt)
    tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab, do_lower_case=True)

    output_dir = args.base_working_path + "/output"
    run_config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_summary_steps=args.save_summary_steps,
        keep_checkpoint_max=1,
        save_checkpoints_steps=args.save_checkpoint_steps,
    )

    bert_config = modeling.BertConfig.from_json_file(bert_config)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(topics),
        init_checkpoint=bert_init_chckpnt,
        learning_rate=args.learning_rate,
        num_train_steps=-1,
        num_warmup_steps=-1,
        use_tpu=False,
        use_one_hot_embeddings=False,
        layer_indexes=args.layer_indices,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=run_config, params={"batch_size": args.batch_size}
    )

    print("=" * 50)
    print(f"Beginning predict")
    print("=" * 50)

    # inference
    embeddings_list = np.empty(
        [len(examples), args.max_seq_length, args.embedding_size]
    )
    label_ids_list = np.empty([len(examples), examples[0].labels.shape[0]])

    generate_bert_embeddings(
        args.mode,
        input_fn,
        embeddings_list,
        label_ids_list,
        estimator,
        tokenizer,
        args.layer_indices,
    )

    print("=" * 50)
    print(f"Embedding list size: {len(embeddings_list)}")
    print(f"Labels list size: {len(label_ids_list)}")
    print(f"Embedding size: {len(embeddings_list[0][0])}, {len(embeddings_list[0])}")
    print(f"Saving...")

    dump_path = f"bert_{args.stage}_{args.mode}.npy"
    np.save(dump_path, embeddings_list)

    dump_path_labels = f"bert_{args.stage}_{args.mode}_labels.npy"
    np.save(dump_path_labels, label_ids_list)

    print("DONE")
    print("=" * 50)
    print(
        f"Finished generating {args.stage} BERT token level embeddings",
        f"in {time()-time_start} seconds.\nPath: {dump_path}",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "stage",
        choices=["train", "dev", "test"],
        help="Predict stage; which data to use",
    )
    parser.add_argument(
        "mode",
        choices=["4x768", "512x768", "4x512x768"],
        help="The specified output embedding dimension",
    )
    parser.add_argument(
        "-df",
        "--dataframe_path",
        help="Location of the Pandas DataFrame containing the preprocessed data",
        default="/home/ianic/dataset_2_with_topics_prot_4.pkl",
    )
    parser.add_argument(
        "-base",
        "--base_path",
        help="Location of the pretrained HierarchicalMultilabelClassifier",
        default="/home/ianic/pretrain/multilingual_L-12_H-768_A-12",
    )
    parser.add_argument(
        "-working",
        "--base_working_path",
        help="Root location of the directory which will contain the data "
        + "in the .tfrecord file format",
        default="/home/ianic/pretrain/working",
    )
    parser.add_argument(
        "-layers",
        "--layer_indices",
        help="Indices of BERT layers which should be taken into account "
        + "when generating embeddings",
        default=[-1, -2, -3, -4],
    )
    parser.add_argument(
        "-seq",
        "--max_seq_length",
        help="The maximum sequence length for BERT",
        default=512,
    )
    parser.add_argument(
        "-emb",
        "--embedding_size",
        help="The BERT embedding length, matches the number of hidden units",
        default=768,
    )
    parser.add_argument("-batch",
        "--batch_size",
        help="Input batch size",
        default=32,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="The learning rate for BERT",
        default=2e-5,
    )
    parser.add_argument(
        "-ckptsteps",
        "--save_checkpoint_steps",
        help="After how much steps a checkpoint should be saved",
        default=1000,
    )
    parser.add_argument(
        "-summsteps",
        "--save_summary_steps",
        help="After how much steps a summary should be saved",
        default=500,
    )
    args = parser.parse_args()

    generate_embeddings(args)


if __name__ == "__main__":
    main()

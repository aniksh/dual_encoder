"""Evaluate the model"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.evaluation import evaluate
from model.input_fn import input_fn
from model.input_fn import load_dataset_from_text
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")

if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the random seed for the whole graph
    tf.set_random_seed(params.seed)

    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build.py".format(json_path)
    params.update(json_path)
    num_oov_buckets = params.num_oov_buckets # number of buckets for unknown words

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Get paths for vocabularies and dataset
    path_vocab = os.path.join(args.data_dir, 'vocab{}'.format(params.min_freq))
    params.vocab_path = path_vocab
    path_test_queries = os.path.join(args.data_dir, 'dev/queries.txt')
    path_test_articles = os.path.join(args.data_dir, 'dev/articles.txt')
    # Load Vocabularies
    vocab = tf.contrib.lookup.index_table_from_file(path_vocab, 
                          num_oov_buckets=num_oov_buckets,
                          key_column_index=0)

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_queries = load_dataset_from_text(path_test_queries, vocab, params)
    test_articles = load_dataset_from_text(path_test_articles, vocab, params)

    # Specify other parameters for the dataset and the model
    params.eval_size = params.test_size
    params.id_pad_word = vocab.lookup(tf.constant(params.pad_word))

    # Create iterator over the test set
    inputs = input_fn('eval', test_queries, test_articles, params)
    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', inputs, params, reuse=False)
    logging.info("- done.")

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)

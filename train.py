"""Train the model"""

import argparse
import logging
import os
from time import time

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn
from model.input_fn import load_dataset_from_text
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='base_model',
          help="Directory containing params.json")
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
          help="Optional, directory containing weights to reload before training")


if __name__ == '__main__':
  args = parser.parse_args()  
  args.model_dir = os.path.join("experiments", args.model_dir)
  if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
  
  # Load the parameters from the experiment params.json file in model_dir
  params = Params('params.json')
  # Load the parameters from the dataset, that gives the size etc. into params
  json_path = os.path.join(args.data_dir, 'dataset_params.json')
  assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
  params.update(json_path)
  
  # Set the random seed for the whole graph for reproductible experiments
  tf.set_random_seed(params.seed)

  # Save the params in model_dir
  json_path = os.path.join(args.model_dir, 'params.json')
  params.save(json_path)
  
  num_oov_buckets = params.num_oov_buckets # number of buckets for unknown words

  # Check that we are not overwriting some previous experiment
  # Comment these lines if you are developing your model and don't care about overwritting
  model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
  overwritting = model_dir_has_best_weights and args.restore_dir is None
  assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

  # Set the logger
  set_logger(os.path.join(args.model_dir, 'train.log'))

  # Get paths for vocabularies and dataset
  path_vocab = os.path.join(args.data_dir, 'vocab{}'.format(params.min_freq))
  params.vocab_path = path_vocab
  path_train_queries = os.path.join(args.data_dir, 'train/queries.txt')
  path_train_articles = os.path.join(args.data_dir, 'train/articles.txt')
  # params.train_size = 10000
  path_eval_queries = os.path.join(args.data_dir, 'dev/queries.txt')
  path_eval_articles = os.path.join(args.data_dir, 'dev/articles.txt')

  # Load Vocabularies
  vocab = tf.contrib.lookup.index_table_from_file(path_vocab, 
                          num_oov_buckets=num_oov_buckets,
                          key_column_index=0)

  # Create the input data pipeline
  logging.info("Creating the datasets...")
  train_queries = load_dataset_from_text(path_train_queries, vocab, params)
  train_articles = load_dataset_from_text(path_train_articles, vocab, params)
  eval_queries = load_dataset_from_text(path_eval_queries, vocab, params)
  eval_articles = load_dataset_from_text(path_eval_articles, vocab, params)

  # Specify other parameters for the dataset and the model
  params.eval_size = params.dev_size
  params.buffer_size = params.train_size # buffer size for shuffling
  params.id_pad_word = vocab.lookup(tf.constant(params.pad_word))

  # Create the two iterators over the two datasets
  train_inputs = input_fn('train', train_queries, train_articles, params)
  eval_inputs = input_fn('eval', eval_queries, eval_articles, params)
  logging.info("- done.")
  # Test input
  # with tf.Session() as sess:
  #   sess.run(tf.tables_initializer())
  #   sess.run(eval_inputs['iterator_init_op'])
  #   start_time = time()
  #   for i in range(200):
  #     sess.run([eval_inputs['query'], eval_inputs['query_lengths']])
  #   print(time() - start_time, "seconds")
  # exit(0)

  # Define the models (2 different set of nodes that share weights for train and eval)
  logging.info("Creating the model...")
  train_model_spec = model_fn('train', train_inputs, params)
  eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)
  logging.info("- done.")

  # Train the model
  logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
  train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_dir)
"""
From Stanford cs230 code examples
https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/nlp/model
"""

"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def load_dataset_from_text(path_txt, vocab, params):
  """Create tf.data Instance from txt file

  Args:
    path_txt: (string) path containing one example per line
    vocab: (tf.lookuptable)
    params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

  Returns:
    dataset: (tf.Dataset) yielding list of ids of tokens for each example
  """
  # Load txt file, one example per line
  dataset = tf.data.TextLineDataset(path_txt)

  # Convert line into list of tokens, splitting by white space
  dataset = dataset.map(lambda string: tf.string_split([string]).values[:params.max_tokens], 
                        num_parallel_calls=params.num_parallel_threads)

  # Lookup tokens to return their ids
  dataset = dataset.map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens)), 
                        num_parallel_calls=params.num_parallel_threads)

  return dataset


def input_fn(mode, queries, articles, params):
  """Input function for LTR

  Args:
    mode: (string) 'train', 'eval' or any other mode you can think of
           At training, we shuffle the data and have multiple epochs
    queries: (tf.Dataset) yielding list of ids of words in queries
    articles: (tf.Dataset) yielding list of ids of words in articles
    params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

  """
  # Load all the dataset in memory for shuffling is training
  is_training = (mode == 'train')
  buffer_size = params.buffer_size if is_training else 1

  # Zip the queries and the articles together
  dataset = tf.data.Dataset.zip((queries, articles))

  # Create batches and pad the queries of different length
  padded_shapes = ((tf.TensorShape([None]),  # query of unknown size
            tf.TensorShape([])),     # size(words)
           (tf.TensorShape([None]),  # articles of unknown size
            tf.TensorShape([])))     # size(tags)

  padding_values = ((params.id_pad_word,   # query padded on the right with id_pad_word
             0),                   # size(words) -- unused
            (params.id_pad_word,    # articles padded on the right with id_pad_tag
             0))                   # size(tags) -- unused


  dataset = (dataset
    .shuffle(buffer_size=buffer_size)
    .padded_batch(params.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
    .prefetch(1)  # make sure you always have one batch ready to serve
  )

  # Create initializable iterator from this dataset so that we can reset at each epoch
  iterator = dataset.make_initializable_iterator()

  # Query the output of the iterator for input to the model
  ((query, query_lengths), (article, article_lengths)) = iterator.get_next()
  init_op = iterator.initializer

  # Build and return a dictionnary containing the nodes / ops
  inputs = {
    'query': query,
    'article': article,
    'query_lengths': query_lengths,
    'article_lengths': article_lengths,        
    'iterator_init_op': init_op
  }

  return inputs

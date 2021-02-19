"""Define the model."""
import logging

import numpy as np
import tensorflow as tf


def build_model(mode, inputs, params):
  """Compute logits of the model (output distribution)

  Args:
    mode: (string) 'train', 'eval', etc.
    inputs: (dict) contains the inputs of the graph (features, labels...)
        this can be `tf.placeholder` or outputs of `tf.data`
    params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

  Returns:
    output: (tf.Tensor) output of the model
  """
  query = inputs['query']  
  article = inputs['article']
  query_lengths = inputs['query_lengths']
  article_lengths = inputs['article_lengths']

  if params.initializer == "pretrained":
    init = tf.constant_initializer(load_embeddings(params))
  else:
    init = tf.random_uniform_initializer(
            -0.5/params.embedding_size, 0.5/params.embedding_size)
  # Get word embeddings for each token in the sentence
  qembeddings = tf.get_variable(name="qembeddings", dtype=tf.float32,
          shape=[params.vocab_size, params.embedding_size],
          initializer=init)
  dembeddings = tf.get_variable(name="dembeddings", dtype=tf.float32,
          shape=[params.vocab_size, params.embedding_size],
          initializer=init)
            
  query_embs = tf.nn.embedding_lookup(qembeddings, query)
  article_embs = tf.nn.embedding_lookup(dembeddings, article)

  # (we need to apply a mask to account for padding)
  query_mask = tf.expand_dims(tf.sequence_mask(query_lengths, dtype=tf.float32), -1)
  article_mask = tf.expand_dims(tf.sequence_mask(article_lengths, dtype=tf.float32), -1)

  if params.model_version == 'embed':
    query_embed = tf.reduce_sum(tf.multiply(query_embs, query_mask), axis=1) \
                    / tf.cast(tf.expand_dims(query_lengths, -1), tf.float32)
    article_embed = tf.reduce_sum(tf.multiply(article_embs, article_mask), axis=1) \
                    / tf.cast(tf.expand_dims(article_lengths, -1), tf.float32)
    
  elif params.model_version == 'embedwt':
    qwts = tf.get_variable(name="qwts", dtype=tf.float32,
          shape=[params.vocab_size], initializer=tf.random_uniform_initializer())
    awts = tf.get_variable(name="awts", dtype=tf.float32,
          shape=[params.vocab_size], initializer=tf.random_uniform_initializer())
    query_wt = tf.nn.embedding_lookup(qwts, query)
    article_wt = tf.nn.embedding_lookup(awts, article)
    query_exp = tf.multiply(tf.exp(tf.expand_dims(query_wt,-1)), query_mask)
    query_softmax = query_exp / tf.reduce_sum(query_exp, axis=1, keepdims=True)
    article_exp = tf.multiply(tf.exp(tf.expand_dims(article_wt,-1)), article_mask)
    article_softmax = article_exp / tf.reduce_sum(article_exp, axis=1, keepdims=True)
    
    query_embed = tf.reduce_sum(tf.multiply(query_embs, query_softmax), axis=1)
    # tf.reduce_sum(tf.multiply(tf.multiply(query_embs, tf.expand_dims(query_wt,-1)), query_mask), axis=1) \
                    # / tf.reduce_sum(tf.multiply(tf.expand_dims(query_wt,-1), query_mask), axis=1)
                    # / tf.cast(tf.expand_dims(query_lengths, -1), tf.float32)                    
    article_embed = tf.reduce_sum(tf.multiply(article_embs, article_softmax), axis=1)
    # tf.reduce_sum(tf.multiply(tf.multiply(article_embs, tf.expand_dims(article_wt,-1)), article_mask), axis=1) \
                    # / tf.reduce_sum(tf.multiply(tf.expand_dims(article_wt,-1), article_mask), axis=1)
                    # / tf.cast(tf.expand_dims(article_lengths, -1), tf.float32)

  elif params.model_version == 'embedtf':
    query_wt = tf.nn.embedding_lookup(params.idf, query)
    article_wt = tf.nn.embedding_lookup(params.idf, article)
    query_exp = tf.multiply(tf.exp(tf.expand_dims(query_wt,-1)), query_mask)
    query_softmax = query_exp / tf.reduce_sum(query_exp, axis=1, keepdims=True)
    article_exp = tf.multiply(tf.exp(tf.expand_dims(article_wt,-1)), article_mask)
    article_softmax = article_exp / tf.reduce_sum(article_exp, axis=1, keepdims=True)
    
    query_embed = tf.reduce_sum(tf.multiply(query_embs, query_softmax), axis=1)
    article_embed = tf.reduce_sum(tf.multiply(article_embs, article_softmax), axis=1)

  elif params.model_version == 'sif':
    vocab_freqs = params.vocab_freqs
    query_wt = 0.001 / (0.001 + tf.nn.embedding_lookup(vocab_freqs, query))
    article_wt = 0.001 / (0.001 + tf.nn.embedding_lookup(vocab_freqs, article))

    query_embed = tf.reduce_sum(tf.multiply(tf.multiply(query_embs, tf.expand_dims(query_wt,-1)), query_mask), axis=1) \
                    / tf.cast(tf.expand_dims(query_lengths, -1), tf.float32)
    article_embed = tf.reduce_sum(tf.multiply(tf.multiply(article_embs, tf.expand_dims(article_wt,-1)), article_mask), axis=1) \
                    / tf.cast(tf.expand_dims(article_lengths, -1), tf.float32)

  # elif params.model_version == 'lstm':
    
  # 	# Apply LSTM over the embeddings
  # 	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
  # 	output, _  = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)

  # 	# Compute logits from the output of the LSTM
  # 	logits = tf.layers.dense(output, params.number_of_tags)

  else:
    raise NotImplementedError("Unknown model version: {}".format(params.model_version))

  return query_embed, article_embed


def model_fn(mode, inputs, params, reuse=False):
  """Model function defining the graph operations.

  Args:
    mode: (string) 'train', 'eval', etc.
    inputs: (dict) contains the inputs of the graph (features, labels...)
        this can be `tf.placeholder` or outputs of `tf.data`
    params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    reuse: (bool) whether to reuse the weights

  Returns:
    model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
  """
  is_training = (mode == 'train')

  # -----------------------------------------------------------
  # MODEL: define the layers of the model
  with tf.variable_scope('model', reuse=reuse):
    # Compute the output distribution of the model and the predictions
    query, article = build_model(mode, inputs, params)

  # Define loss
  batch_size = tf.shape(article)[0]  
  logit_matrix = tf.matmul(query, article, transpose_b=True)
  pos_logits = tf.diag_part(logit_matrix)
  pos_labels = tf.ones_like(pos_logits)
  neg_positions = 1 - tf.diag(pos_labels)
  neg_logits = tf.boolean_mask(logit_matrix, neg_positions)
  neg_labels = tf.zeros_like(neg_logits)
  # labels = tf.diag(tf.ones_like(tf.diag_part(logit_matrix)))
  # losses = - tf.log(diag_matrix) \
  #           - tf.log(1 - (logit_matrix - diag_matrix))
  if params.loss_fn == "xent":
    pos_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=pos_labels, logits=pos_logits)
    neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=neg_labels, logits=neg_logits)
    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    loss = (tf.reduce_mean(pos_xent) + tf.reduce_mean(neg_xent)) / 2
  elif params.loss_fn == "margin":
    neg_logits = tf.reduce_max(logit_matrix - tf.diag(np.inf * pos_labels), axis=1)
    loss = tf.reduce_mean(tf.maximum(0., 0.5 - tf.sigmoid(pos_logits) + tf.sigmoid(neg_logits)))
  # query = tf.nn.l2_normalize(query)
  # article = tf.nn.l2_normalize(article)
  eval_logits = tf.matmul(query, article, transpose_b=True)
  eval_ranks = tf.argmax(eval_logits, axis=1, output_type=tf.int32)
  precision1 = tf.reduce_mean(tf.cast(tf.equal(eval_ranks, tf.range(batch_size)), tf.float32))
  predictions = tf.cast(eval_logits > 0, tf.float32)
  # accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
  pos_pred = tf.diag_part(predictions)
  neg_pred = tf.boolean_mask(predictions, neg_positions)
  pos_acc = tf.reduce_mean(tf.cast(tf.equal(pos_labels, pos_pred), tf.float32))
  neg_acc = tf.reduce_mean(tf.cast(tf.equal(neg_labels, neg_pred), tf.float32))
  accuracy = (pos_acc + neg_acc) / 2

  
  # Define training step that minimizes the loss with the Adam optimizer
  if is_training:
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    # optimizer = tf.train.AdagradOptimizer(params.learning_rate)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

  # -----------------------------------------------------------
  # METRICS AND SUMMARIES
  # Metrics for evaluation using tf.metrics (average over whole dataset)
  with tf.variable_scope("metrics"):
    metrics = {
      'loss': tf.metrics.mean(loss),
      'pos_acc': tf.metrics.accuracy(labels=pos_labels, predictions=pos_pred),
      'neg_acc': tf.metrics.accuracy(labels=neg_labels, predictions=neg_pred),
      'accuracy': tf.metrics.mean([pos_acc, neg_acc]),
      'precision1': tf.metrics.mean(precision1)
    }

  # Group the update ops for the tf.metrics
  update_metrics_op = tf.group(*[op for _, op in metrics.values()])

  # Get the op to reset the local variables used in tf.metrics
  metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
  metrics_init_op = tf.variables_initializer(metric_variables)

  # Summaries for training
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('pos_acc', pos_acc)
  tf.summary.scalar('neg_acc', neg_acc)
  tf.summary.scalar('accuracy', accuracy)
  tf.summary.scalar('precision1', precision1)

  # -----------------------------------------------------------
  # MODEL SPECIFICATION
  # Create the model specification and return it
  # It contains nodes or operations in the graph that will be used for training and evaluation
  model_spec = inputs
  variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
  model_spec['variable_init_op'] = variable_init_op
  model_spec["predictions"] = predictions
  model_spec['loss'] = loss
  model_spec['accuracy'] = accuracy
  model_spec['precision1'] = precision1
  model_spec['metrics_init_op'] = metrics_init_op
  model_spec['metrics'] = metrics
  model_spec['update_metrics'] = update_metrics_op
  model_spec['summary_op'] = tf.summary.merge_all()

  if is_training:
    model_spec['train_op'] = train_op

  return model_spec


def load_embeddings(params):
  logging.info("Loading pretrained embeddings from %s..." % params.pretrained_path)
  word_to_index = {}
  with open(params.vocab_path) as f:
    for i, line in enumerate(f):
      word_to_index[line.strip().split()[0]] = i
  
  embed = np.zeros((len(word_to_index) + 1, params.embedding_size))
  np.random.seed(params.seed)
  embed[-1] = np.random.uniform(-0.5/params.embedding_size, 
                                0.5/params.embedding_size, size=(params.embedding_size,))

  with open(params.pretrained_path) as f:
    if "glove" not in params.pretrained_path:
      _, _ = f.readline().strip().split()
    for line in f:
      word, numbers = line.strip().split(maxsplit=1)
      word_ind = word_to_index.get(word, -1)
      if word_ind > 0:
        embed[word_ind] = np.fromstring(numbers, count=params.embedding_size, sep=' ')
  
  logging.info("Done")
  return embed

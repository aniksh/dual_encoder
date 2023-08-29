# Dual Encoder

The code structure is based on https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/nlp

## Dataset

The signal media 1 million news articles data set can be obtained from https://research.signal-ai.com/datasets/signal1m.html
Extract the *jsonl* file and move to data *data* directory to process the data set.

`build_dataset.py` processes the dataset for training.  
```bash
python build_dataset.py news
```

Then `build_vocab.py` is used to create vocabulary file.
```bash
python build_vocab.py --data_dir data/signal_news --min_freq 10
```

## Training

```bash
python train.py --data_dir DATA --model_dir EXP_NAME --restore_dir WEIGHTS_DIR
```
  - DATA is the path to data set, *data/signal_news*
  - EXP_NAME is a name for the training run
  - WEIGHTS_DIR is the directory containing model weights e.g. EXP_NAME/best_weights

The settings for training are in the *params.json* file.

Description:

  - "max_tokens": Maximum number of tokens in a query or articles (larger articles are truncated)
  - "num_parallel_threads": Number of parallel threads for input data pipeline (should be less
  than the total number of cores in the system)
  - "seed": Random seed (int)
  - "model_version": Default is 'embed' (Only model till now, other models have not been implemented)
  - "embedding_size": Word embedding dimension (int)
  - "initializer": Word embedding initializer (random or pretrained)
  - "pretrained_path": Path to the pretrained glove embedding file

  - "loss_fn": Loss function (w2v or margin)
  - "learning_rate": Learning rate (default: 1e-3)
  - "batch_size": Batch size (e.g. 100, 500 or 1000)
  - "num_epochs": Number of epochs
  - "tol": If validation precision1 does not improve in tol epochs, stop training
  - "gpu": ID of gpu (default: "0")

  - "save_summary_steps": Save loss and other metrics after this many steps


## Ranking

```
python rank.py --data_dir DATA --model EXP_NAME --method METHOD --mode MODE
```

Ranks the test set using weights from the saved model.

There are 5 options for the *method* argument
 - *embed* uses the Bag-of-Embedding method to obtain embeddings for queries and articles
 - *sif* uses the [SIF](https://github.com/PrincetonML/SIF) (Smooth Inverse Frequence weighting scheme) method
 - *tfidf* uses the TF-IDF ranking method
 - *lm* uses pretrained language models to obtain sentence embeddings with *sentence_transformers* library
 - *use* uses the universal sentence encoder model to embed sentences

For the Bag-of-Embedding method, we experimented with different modes of ranking. The *mode* argument can be used for this
 - *rank* if for ranking using the dot product of query and article embeddings
 - *ranktf* adds the TF-IDF score to the dot product for ranking
 - *rerank* first ranks the articles using TF-IDF scores and then re-ranks the first 1000 articles from the ranked list
 - *reranktf* adds TF-IDF scores when re-ranking the first 1000 results from TF-IDF method 

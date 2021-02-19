# Dual Encoder

The code structure is based on https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/nlp

## Training

```
python train.py --model_dir embed-rand-d100-b100-e20-margin
```

The settings for training are in the *params.json* file.

Description:

  - "max_tokens": Maximum number of tokens in a query or articles (larger articles are truncated)
  - "num_parallel_threads": Number of parallel threads for input data pipeline (should be less
  than the total number of cores in the system)
  - "seed": Random seed (int)
  - "model_version": Default is 'embed' (Only model till now, other models have not been implemented)
  - "embedding_size": Word embedding dimension (int)
  - "initializer": Word embedding initializer (random or pretrained)
  - "pretrained_path": Path to the pretrained embedding file

  - "loss_fn": Loss function (w2v or margin)
  - "learning_rate": Learning rate (default: 1e-3)
  - "batch_size": Batch size (e.g. 100, 500 or 1000)
  - "num_epochs": Number of epochs
  - "tol": If validation precision1 does not improve in tol epochs, stop training
  - "gpu": ID of gpu (default: "0")

  - "save_summary_steps": Save loss and other metrics after this many steps


## Ranking

```
python rank.py --model experiments/embed-rand-d100-b100-e20-margin/best_weights
```

Ranks the test set using weights from the saved model.
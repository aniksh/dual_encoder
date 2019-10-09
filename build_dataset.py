import json
import os
from time import time
from nltk import word_tokenize
import numpy as np

def load_dataset(load_path):
  queries = []
  articles = []
  start = time()
  with open(load_path) as f:
    for i, line in enumerate(f):
      if (i+1) % 1000 == 0:
        interval = time() - start
        print("Read {:>,10d} lines, {:>5.2f} lines/sec".format(i+1, (i+1) / interval), end="\r")
      # if i == 3000:
      #     break
      element = json.loads(line)
      query_tokens = word_tokenize(element['title'].lower())
      doc_tokens = word_tokenize(element['content'].lower())
      queries.append(" ".join(query_tokens))
      articles.append(" ".join(doc_tokens))

  return queries, articles

def save_dataset(save_dir, dataset, indices):
  # Create directory if it doesn't exist
  print("Saving in {}...".format(save_dir))
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  queries, articles = dataset
  with open(os.path.join(save_dir, "queries.txt"), 'w') as fq:
    with open(os.path.join(save_dir, "articles.txt"), 'w') as fd:
      for i in indices:
        fq.write(queries[i] + "\n")
        fd.write(articles[i] + "\n")
  return

if __name__ == "__main__":
  # Load dataset into memory
  print("Loading the dataset...")
  data_path = '../data/signalmedia-1m.jsonl'
  dataset = load_dataset(data_path)
  print("Done\n")

  dev_size = 10_000
  test_size = 10_000

  # Shuffle the dataset indices to split
  data_indices = np.random.permutation(np.arange(len(dataset[0])))
  train_indices = data_indices[(dev_size + test_size):]
  dev_indices = data_indices[:dev_size]
  test_indices = data_indices[dev_size: (dev_size + test_size)]

  save_dataset("data/train", dataset, train_indices)
  save_dataset("data/dev", dataset, dev_indices)
  save_dataset("data/test", dataset, test_indices)

  print("Done")
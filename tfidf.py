import pickle
import os
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_by_line(path_to_file, max_lines=-1):
  lines = []
  with open(path_to_file) as f:
    for i, line in enumerate(f):
      lines.append(line.rstrip())
      if i == max_lines - 1:
        break
  return lines

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("data_dir", default="data", help="Directory containing the dataset")
  args = parser.parse_args()
  args.data_dir = os.path.join("data", args.data_dir)

  print("Loading articles text...")
  articles = load_by_line(os.path.join(args.data_dir, 'test/articles.txt')) #+\
            #  load_by_line(os.path.join(args.data_dir, 'dev/articles.txt')) +\
            #  load_by_line(os.path.join(args.data_dir, 'train/articles.txt'))
  print("Done.\n")

  # vocab = {}
  # with open(os.path.join(args.data_dir, "vocab100")) as f:
  #   for i, line in enumerate(f):
  #     vocab[line.split()[0]] = i
  
  # print(f"Vocab size: {len(vocab)}")
  # tfidf = TfidfVectorizer(vocabulary=vocab)

  tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), sublinear_tf=True, dtype=np.float32,)
  print("Fitting TfIdf...")
  tfidf.fit(articles)
  print("Done.\n")
  print("Saving model")
  with open(os.path.join(args.data_dir, "tfidfvec_test2.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
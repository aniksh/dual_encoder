from collections import Counter, OrderedDict
import json
from argparse import ArgumentParser
import os

# Hyper parameters for the vocab
NUM_OOV_BUCKETS = 1 # number of buckets (= number of ids) for unknown words
PAD_WORD = '<pad>'

def get_args():
  parser = ArgumentParser(description="Builds "
                "the vocabulary from the dataset")
  
  parser.add_argument("--data_dir", default="data", 
                      help="Directory containing the data")
  parser.add_argument("--min_freq", default=1, 
                      type=int,
                      help="Minimum word frequency")
  
  return parser.parse_args()


def update_vocab(text_file, vocab):
  """
  Assume the text_file has tokenized words 
  separated by space in each line

  Args:
    text_file: text file path
    vocab: dictionary (Counter)
  """
  print("Reading {}...".format(text_file))
  with open(text_file) as f:
    for i, line in enumerate(f):
      vocab.update(line.strip().split())
  return i + 1

def save_vocab_to_text(vocab_file, vocab):
  """
  Write vocabulary words to text file

  Args:
    vocab_file: save file path
    vocab: dictionary {word:freq}
  """
  print("Writing to {}...".format(vocab_file))
  with open(vocab_file, 'w') as f:
    for word, freq in vocab:
      f.write("{}\t{}\n".format(word,freq))
  return

def save_dict_to_json(d, json_path):
  """Saves dict to json file
  Args:
    d: (dict)
    json_path: (string) path to json file
  """
  with open(json_path, 'w') as f:
    d = {k: v for k, v in d.items()}
    json.dump(d, f, indent=4)

if __name__ == "__main__":
  args = get_args()

  words = Counter()

  print("Building the vocabulary...")
  size_train = update_vocab(os.path.join(args.data_dir, "train/queries.txt"), words)
  size_dev = update_vocab(os.path.join(args.data_dir, "dev/queries.txt"), words)
  # size_test = update_vocab(os.path.join(args.data_dir, "test/queries.txt"), words)
  _ = update_vocab(os.path.join(args.data_dir, "train/articles.txt"), words)
  _ = update_vocab(os.path.join(args.data_dir, "dev/articles.txt"), words)
  # _ = update_vocab(os.path.join(args.data_dir, "test/articles.txt"), words)
  print("Done\n")

  # Sort vocab by freq and only keep words with min_freq count
  vocab = [(PAD_WORD, 0)]
  vocab += [(w,f) for (w,f) in words.most_common() if f >= args.min_freq]
  
  # Save vocabulary
  print("Saving the vocabulary...")  
  save_vocab_to_text(os.path.join(args.data_dir, "vocab{}".format(args.min_freq)), vocab)
  print("Done\n")

  # Save datasets properties in json file
  sizes = {
    'train_size': size_train,
    'dev_size': size_dev,
    # 'test_size': size_test,
    'min_freq': args.min_freq,
    'vocab_size': len(vocab) + NUM_OOV_BUCKETS,
    'pad_word': PAD_WORD,
    'num_oov_buckets': NUM_OOV_BUCKETS
  }
  save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

  # Logging sizes
  to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
  print("Characteristics of the dataset:\n{}".format(to_print))
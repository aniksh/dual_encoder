import json
import os
import sys
from time import time
from nltk import word_tokenize, sent_tokenize
import numpy as np

def load_dataset(load_path, max_lines=-1, mode='news'):
  queries = []
  articles = []
  start = time()
  count = 0
  skip_lines = 0
  line_nos = []
  with open(load_path, encoding='utf-8') as f:
    for i, line in enumerate(f):
      if (i+1) % 100 == 0:
        interval = time() - start
        print("Read {:>10,d} lines, {:>5.2f} lines/sec".format(i+1, (i+1) / interval), end="\r")
      
      element = json.loads(line)
      if mode == 'news':
        query_tokens = word_tokenize(element['title'].lower())
        doc_tokens = word_tokenize(element['content'].lower())
        line_nos.append(i)
      elif mode == 'news-sent':
        sents = sent_tokenize(element['content'].lower())
        query_tokens = word_tokenize(sents[0].lower())
        doc_tokens = [token for sent in sents[1:] for token in word_tokenize(sent.lower(), preserve_line=True)]

        if len(query_tokens) < 5 or len(doc_tokens) < 5:
          skip_lines += 1
          continue
        else:
          line_nos.append(i)
      
      elif mode == 'wiki':
        query_tokens = word_tokenize(element['title'].lower())
        doc_tokens = word_tokenize(element['text'].lower())
        line_nos.append(i)
      elif mode == 'wiki-sent':
        sents = sent_tokenize(element['text'].lower())
        try:
          _, text = sents[0].split('\n\n', 1)
        except:
          skip_lines += 1
          continue
        
        query_tokens = word_tokenize(text.lower())
        doc_tokens = [token for sent in sents[1:] for token in word_tokenize(sent.lower(), preserve_line=True)]

        if len(query_tokens) < 5 or len(doc_tokens) < 5:
          skip_lines += 1
          continue
        else:
          line_nos.append(i)
      
      elif mode == 'wiki-para':
        query_tokens = word_tokenize(element['title'].lower())
        try:
          _, para, __ = element['text'].split('\n\n',2)
        except:
          skip_lines += 1
          continue
        
        doc_tokens = word_tokenize(para.lower())
        if len(doc_tokens) < 5:
          skip_lines += 1
          continue
        else:
          line_nos.append(i)

      queries.append(" ".join(query_tokens))
      articles.append(" ".join(doc_tokens))
      count += 1

      if count == max_lines:
          break
  print("\nSkipped {} lines".format(skip_lines))

  return queries, articles, np.array(line_nos)

def save_dataset(save_dir, dataset, indices):
  # Create directory if it doesn't exist
  print("Saving in {}...".format(save_dir))
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  queries, articles, line_nos = dataset
  np.savetxt(os.path.join(save_dir, "indices.txt"), line_nos[indices], fmt='%d')
  
  with open(os.path.join(save_dir, "queries.txt"), 'w', encoding='utf-8') as fq:
    with open(os.path.join(save_dir, "articles.txt"), 'w', encoding='utf-8') as fd:
      for i in indices:
        fq.write(queries[i] + "\n")
        fd.write(articles[i] + "\n")
  return

if __name__ == "__main__":
  dataset_name = sys.argv[1] # news, wiki or wiki-sent
  # Load dataset into memory
  print("Loading the dataset...")
  if 'news' in dataset_name:
    data_path = '../data/signalmedia-1m.jsonl'
  elif 'wiki' in dataset_name:
    data_path = '../data/wiki-json/all_docs.jsonl'
  
  dataset = load_dataset(data_path, max_lines=-1, mode=dataset_name)
  print("Done\n")

  dev_size = 10000
  test_size = 10000

  # Shuffle the dataset indices to split
  np.random.seed(1)
  data_indices = np.random.permutation(len(dataset[2]))
  train_indices = data_indices[(dev_size + test_size):]
  dev_indices = data_indices[:dev_size]
  test_indices = data_indices[dev_size: (dev_size + test_size)]

  if dataset_name == 'news':
    save_path = 'data/signal_news'
  else:
    save_path = 'data/' + dataset_name
  
  save_dataset(save_path + "/train", dataset, train_indices)
  save_dataset(save_path + "/dev", dataset, dev_indices)
  save_dataset(save_path + "/test", dataset, test_indices)

  print("Done")
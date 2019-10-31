import json
import glob
from collections import defaultdict
import argparse
import os
from nltk import word_tokenize

import numpy as np

from load_embed import Embedding, cosine_sim

def load_data():
    driver2relevance = defaultdict(list)
    driver2articles = defaultdict(list)
    
    for i in range(0, 6):
        for f in glob.glob('../data/benchmark/eval_files/eval_' + str(i) + '_*'):
            with open(f) as input_f:
                # print(f)
                evl = json.load(input_f)
                for driver in evl:
                    ratings = []
                    if driver not in driver2relevance:
                        new_ratings = True
                    else:
                        old_ratings = driver2relevance[driver]
                        new_ratings = False
                    if driver in driver2articles:                      
                        new_articles = False
                    else:
                        new_articles = True
        
                    docs = evl[driver]
                    for index, doc in enumerate(docs):
                        relevance = doc['relevant']
                        if new_articles:
                          article = word_tokenize(doc['title'].lower()) + word_tokenize(doc['abstract'].lower())
                          driver2articles[driver].append(article)
                        if new_ratings:
                          ratings.append(relevance)
                        else:
                          ratings.append((old_ratings[index] + relevance)/2)
                        
                    driver2relevance[driver] = ratings
      
    return driver2articles, driver2relevance

class GloveEmbedder():
  def __init__(self, save_path, embed_dim):
    self.save_path = save_path
    self.embed_dim = embed_dim
    self.load_vocab()
    self.load_embed()

  def load_vocab(self):
    id2word = [line.split()[0]
        for line in open(os.path.join("data", "vocab100"), 'r')
        ]
    self.id2word = id2word
    word2id = {}
    for _i in range(len(id2word)):
      word2id[id2word[_i]] = _i
    self.word2id = word2id    
  
  def load_embed(self):
    self.embed = np.zeros((len(self.id2word), self.embed_dim))
    with open(self.save_path) as f:
      # _, _ = f.readline().strip().split()
      for line in f:
        word, numbers = line.strip().split(maxsplit=1)
        if word in self.id2word:
          self.embed[self.word2id[word]] = np.fromstring(numbers, sep=' ')
  
  def get_query_embed(self, word_list, norm=True):
    mean_emb = np.mean([self.embed[self.word2id[word]] for word in word_list if word in self.word2id], axis=0)
    if norm:
      return mean_emb / np.linalg.norm(mean_emb)
    else:
      return mean_emb

  def get_article_embed(self, word_list, norm=True):
    mean_emb = np.mean([self.embed[self.word2id[word]] for word in word_list if word in self.word2id], axis=0)
    if norm:
      return mean_emb / np.linalg.norm(mean_emb)
    else:
      return mean_emb

def rank(qemb, aembs, num=5):
  adist = np.dot(aembs, qemb.T)
  highsim_idxs = adist.argsort()[::-1]
  highsim_idxs = highsim_idxs[:num]
  return highsim_idxs

def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, help="Model path")
  # parser.add_argument("--epoch", type=int, default=-1, help="Model epoch")
  args = parser.parse_args()

  q2a, q2r = load_data()
  # _start_shell()
  # exit()
  # embed = Embedding(save_path=args.model)
  print("Loading Glove...")
  embed = GloveEmbedder("/home/anik/Documents/f18/embeddings/data/glove.6B/glove.6B.50d.txt")
  print("Done.")

  ratings = []
  for q in q2a:
    articles_embed = np.array([embed.get_article_embed(a) for a in q2a[q]])
    query_embed = embed.get_query_embed(q.split())
    ratings += [q2r[q][i] for i in rank(query_embed, articles_embed, 3)]
  
  print("\nMean Rank:", np.mean(ratings))
  
  
  # while True:
  #   try:
  #     text = input("\nEnter query index\t")
  #     try:
  #       example = queries[int(text)]
  #     except:
  #       example = text.split()
  #     query_embed = embed.get_query_embed(example)
      
  #     print("Query:", " ".join(example))
  #     print("Aritcles:")
  #     for r, i in enumerate(rank(query_embed, articles_embed)):
  #       print("Rank %d: Index %d -" %(r+1, i), " ".join(articles[i]), "\n")

  #   except KeyboardInterrupt:
  #     print()
  #     break
    
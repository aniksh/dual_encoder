import os
import sys
import argparse
import tensorflow as tf
import numpy as np

def cosine_sim(a,b):
  dot_product = np.dot(a, b)
  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  return dot_product / (norm_a * norm_b)

class Embedding(object):
  def __init__(self, save_path, data_path, mode="embed", epoch=None, ckpt_file=None, verbose=True):
    # create a new session and a new graph every time this object is constructed
    # if a ckpt file is not provided, use the latest ckpt file.
    if epoch is None or epoch < 0:
      self.ckpt_file = ckpt_file
    else:
      self.ckpt_file = os.path.join(save_path, 'after-epoch-%d' %(epoch))
    self.save_path = save_path
    self.data_path = data_path
    self.mode = mode
    self.epoch = epoch
    with tf.Session() as session:
      self.session = session
      self.load_model(verbose)
      self.load_vocab()

  def load_vocab(self):
    id2word, id2freq = list(zip(*[line.split()
        for line in open(os.path.join(self.data_path, "vocab100"), 'r')
        ]))
    assert len(id2word) + 1 == self.vocab_size, \
            'Expecting vocab size to match ckpt:{} vocab.txt{}'.format(self.vocab_size, len(id2word))
    self.id2word = id2word
    word2id, word2freq = {}, {}
    id2freq = [float(f) for f in id2freq]
    total_freq = np.sum(id2freq)
    for _i in range(len(id2word)):
      word2id[id2word[_i]] = _i
      word2freq[id2word[_i]] = id2freq[_i] / total_freq
    self.word2id = word2id
    self.word2freq = word2freq
    
  def load_model(self, verbose=True):
    latest_ckpt_file = tf.train.latest_checkpoint(self.save_path) if self.ckpt_file is None else self.ckpt_file
    if verbose and self.ckpt_file is None:
      print('Using the latest checkpoint file', latest_ckpt_file)
    elif verbose:
      print('Using the provided checkpoint file: ', self.ckpt_file)

    meta_graph_path = latest_ckpt_file + '.meta'
    new_saver = tf.train.import_meta_graph(meta_graph_path)
    new_saver.restore(self.session, latest_ckpt_file)

    [query_embs, doc_embs] = self.session.run(['model/qembeddings:0', 'model/dembeddings:0'])
    if self.mode == "embedwt" or self.mode == "embedwt2":
      [self.qwts, self.awts] = self.session.run(['model/qwts:0', 'model/awts:0'])
    self.vocab_size = query_embs.shape[0]
    self.word_dim = query_embs.shape[1]

    self.query_embs = query_embs.copy()
    self.query_embs_n = query_embs / np.linalg.norm(query_embs, axis=-1, keepdims=True)
    self.doc_embs = doc_embs.copy()
    self.doc_embs_n = doc_embs / np.linalg.norm(doc_embs, axis=-1, keepdims=True)

  def get_query_embed(self, word_list, norm=True):
    if self.mode == "sif":
      mean_emb = np.mean([self.query_embs[self.word2id.get(word, -1)] * (0.001 / (0.001 + self.word2freq.get(word, 0.))) for word in word_list], axis=0)
    elif self.mode == "embedwt":
      mean_emb = np.mean([self.query_embs[self.word2id.get(word, -1)] * self.qwts[self.word2id.get(word, -1)] for word in word_list], axis=0)
    elif self.mode == "embedwt2":
      wts = [np.exp(self.qwts[self.word2id.get(word, -1)]) for word in word_list]
      embs = [self.query_embs[self.word2id.get(word, -1)] for word in word_list]
      mean_emb = np.sum(np.multiply(embs, wts), axis=0) / np.sum(wts)
    else:
      mean_emb = np.mean([self.query_embs[self.word2id.get(word, 0)] for word in word_list], axis=0)
    if norm:
      return mean_emb / np.linalg.norm(mean_emb)
    else:
      return mean_emb
  
  def get_article_embed(self, word_list, norm=True):
    if self.mode == "sif":
      mean_emb = np.mean([self.doc_embs[self.word2id.get(word, -1)] * (0.001 / (0.001 + self.word2freq.get(word, 0.))) for word in word_list], axis=0)
    elif self.mode == "embedwt":
      mean_emb = np.mean([self.doc_embs[self.word2id.get(word, -1)] * self.awts[self.word2id.get(word, -1)] for word in word_list], axis=0)
    elif self.mode == "embedwt2":
      wts = [np.exp(self.awts[self.word2id.get(word, -1)]) for word in word_list]
      embs = [self.doc_embs[self.word2id.get(word, -1)] for word in word_list]
      mean_emb = np.sum(np.multiply(embs, wts), axis=0) / np.sum(wts)
    else:
      # word_ids = [self.word2id.get(word, 0) for word in word_list]
      # mean_emb = np.mean([self.doc_embs[i] for i in word_ids if i > 0], axis=0)
      mean_emb = np.mean([self.doc_embs[self.word2id.get(word, 0)] for word in word_list], axis=0)
    if norm:
      return mean_emb / np.linalg.norm(mean_emb)
    else:
      return mean_emb
  
  def get_articles_embed(self, article_list, norm=True, split=True):
    articles_emb = np.zeros((len(article_list), self.word_dim))
    for i, word_list in enumerate(article_list):
      if split:
        word_list = word_list.split()
      if self.mode == "sif":
        articles_emb[i] = np.mean([self.doc_embs[self.word2id.get(word, -1)] * (0.001 / (0.001 + self.word2freq.get(word, 0.))) for word in word_list], axis=0)
      elif self.mode == "embedwt":
        articles_emb[i] = np.mean([self.doc_embs[self.word2id.get(word, -1)] * self.awts[self.word2id.get(word, -1)] for word in word_list], axis=0)
      elif self.mode == "embedwt2":
        wts = [np.exp(self.awts[self.word2id.get(word, -1)]) for word in word_list]
        embs = [self.doc_embs[self.word2id.get(word, -1)] for word in word_list]
        articles_emb[i] = np.sum(np.multiply(embs, wts), axis=0) / np.sum(wts)
      else:
        word_ids = [self.word2id.get(word, -1) for word in word_list]
        articles_emb[i] = np.mean([self.doc_embs[word_id] for word_id in word_ids if word_id >= 0], axis=0)
    if norm:
      return articles_emb / np.linalg.norm(articles_emb, axis=-1, keepdims=True)
    else:
      return articles_emb

  def nearby(self, word, dic="query", num_nns=10):
    assert word in self.word2id, "Word is not in the vocabulary"

    idx = self.word2id[word]
    if dic=="query":
      word_emb = self.query_embs_n[idx]
    elif dic=="doc":
      word_emb = self.doc_embs_n[idx]
    else:
      print("Either query or doc")
      return
    qdist = np.dot(self.query_embs_n, word_emb.T)
    ddist = np.dot(self.doc_embs_n, word_emb.T)
    # print("Top 10 highest similarity for %s" %word)
    highsim_idxs = qdist.argsort()[::-1]
    # select top num_nns (linear) indices with the highest cosine similarity
    highsim_idxs = highsim_idxs[1:num_nns+1]
    words = [self.id2word[j] for j in highsim_idxs]
    print("Nearest neighbours in query embeddings")
    print(" ".join(words))
    
    # print("Top 10 highest similarity for %s" %word)
    highsim_idxs = ddist.argsort()[::-1]
    # select top num_nns (linear) indices with the highest cosine similarity
    highsim_idxs = highsim_idxs[1:num_nns+1]
    words = [self.id2word[j] for j in highsim_idxs]
    print("Nearest neighbours in document embeddings")
    print(" ".join(words))
  
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
  parser.add_argument("model", type=str, help="Model path")
  # parser.add_argument("--epoch", type=int, default=-1, help="Model epoch")
  args = parser.parse_args()
  embed = Embedding(save_path=args.model)
  # _start_shell(locals())
  # sys.exit()
  while True:
    try:
      text = input("\nEnter word\t")
      if "," in text:
        word, dic = text.split(",")
      else:
        word, dic = text, "query"
      embed.nearby(word, dic.strip())
    except KeyboardInterrupt:
      print()
      break
    except AssertionError as e:
      print(e)
    

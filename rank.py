import argparse
import os
import sys
import time, datetime

import numpy as np

# from benchmark_rank import GloveEmbedder

def cosine_sim(a,b):
  dot_product = np.dot(a, b)
  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  return dot_product / (norm_a * norm_b)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def load_by_line(path_to_file):
  lines = []
  with open(path_to_file) as f:
    for line in f:
      lines.append(line.strip().split())
  return lines

def rank(qemb, aembs, num=5):
  adist = np.dot(aembs, qemb.T)
  highsim_idxs = adist.argsort()[::-1]
  highsim_idxs = highsim_idxs[:num]
  return highsim_idxs, adist[highsim_idxs]

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
  parser.add_argument("--data_dir", default="data", help="Directory containing the dataset")
  parser.add_argument("--method", default="embed", help="Embedding method (embed, bert, use)")
  # parser.add_argument("--epoch", type=int, default=-1, help="Model epoch")
  args = parser.parse_args()
  if args.method == "embed":
    from load_embed import Embedding
    embed = Embedding(save_path=args.model)
  elif args.method == "bert":
    from sentence_transformers import SentenceTransformer
    embed = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
  elif args.method == "use":
    import tensorflow_hub as hub
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(module_url)
  # print("Loading Glove...")
  # embed = GloveEmbedder("/home/anik/Documents/f18/embeddings/data/glove.6B/glove.6B.100d.txt", 100)
  # print("Done.")
  
  path_eval_queries = os.path.join(args.data_dir, 'test/queries.txt')
  path_eval_articles = os.path.join(args.data_dir, 'test/articles.txt')

  print("Loading queries and articles text...")
  queries = load_by_line(path_eval_queries)#[:100]
  articles = load_by_line(path_eval_articles)#[:100]
  print("Done.\n")

  print("Loading article embeddings...")
  t1 = time.time()
  if args.method == "embed":
    articles_embed = np.array([embed.get_article_embed(a, norm=False) for a in articles])
  elif args.method == "bert":
    articles_embed = np.array(embed.encode([" ".join(a) for a in articles], batch_size=200, show_progress_bar=True))
  elif args.method == "use":
    import tensorflow as tf
    config = tf.ConfigProto()
    config.graph_options.rewrite_options.shape_optimization = 2
    session = tf.Session(config=config)
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    articles_embed = session.run(embed([" ".join(a) for a in articles]))
  
  articles_embed /= np.linalg.norm(articles_embed, axis=-1, keepdims=True)
  print("Done in {}, shape={}\n".format(datetime.timedelta(seconds=time.time() - t1),
                                         articles_embed.shape))

  rr = np.zeros((len(queries),))
  precision1 = np.zeros((len(queries),))
  precision3 = np.zeros((len(queries),))
  for i, q in enumerate(queries):
    if args.method == "embed":
      query_embed = embed.get_query_embed(q, norm=False)
    elif args.method == "bert":
      query_embed = embed.encode([" ".join(q)]).squeeze()
    elif args.method == "use":
      query_embed = session.run(embed([" ".join(q)])).squeeze()

    query_embed /= np.linalg.norm(query_embed, axis=-1, keepdims=True)
    ids, dots = rank(query_embed, articles_embed, num=None)
    precision1[i] = float(ids[0] == i)
    precision3[i] = float(i in ids[:3])
    r = np.where(ids==i)
    rr[i] = 1 / (r[0][0]+1)
  
  if args.method == "use":
    session.close()
  
  print("Mean Reciprocal Rank:", np.mean(rr))
  print("Precision@1:", np.mean(precision1))
  print("Precision@3:", np.mean(precision3))
  print("Total time: {}".format(datetime.timedelta(seconds=time.time() - t1)))
    
  sys.exit(0)

  while True:
    try:
      text = input("\nEnter query index\t")
      try:
        index = int(text)
        example = queries[index]
      except:
        example = text.split()
      query_embed = embed.get_query_embed(example, norm=True)
      
      print("Query:", " ".join(example))
      print("Actual Article:", " ".join(articles[index]))
      print("\nAritcles:")
      ids, dots = rank(query_embed, articles_embed, num=10)
      for r in range(len(ids)):
        print("Rank %d: Index %d, prob = %.2f" %(r+1, ids[r], sigmoid(dots[r])), " ".join(articles[ids[r]]), "\n")

    except KeyboardInterrupt:
      print()
      break
    
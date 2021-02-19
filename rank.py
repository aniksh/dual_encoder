import argparse
import os
import sys
import pickle
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

def load_by_line(path_to_file, max_lines=-1):
  lines = []
  with open(path_to_file, encoding='utf-8') as f:
    for i, line in enumerate(f):
      lines.append(" ".join(line.strip().split()[:1000]))
      if i == max_lines - 1:
        break
  return lines

def rank(qemb, aembs, num=5):
  adist = np.dot(aembs, qemb.T)
  if args.method == "tfidf":
    adist = adist.toarray().squeeze()
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
  parser.add_argument("--model", default="", type=str, help="Model path")
  parser.add_argument("--data_dir", default="data", help="Directory containing the dataset")
  parser.add_argument("--method", default="embed", help="Embedding method (embed, bert, use)")
  parser.add_argument("--mode", default="rank", help="Ranking mode (rank or rerank)")
  # parser.add_argument("--epoch", type=int, default=-1, help="Model epoch")
  args = parser.parse_args()
  # print("Loading Glove...")
  # embed = GloveEmbedder("/home/anik/Documents/f18/embeddings/data/glove.6B/glove.6B.100d.txt", 100)
  # print("Done.")
  args.model = os.path.join("experiments", args.data_dir, args.model, "best_weights")
  args.data_dir = os.path.join("data", args.data_dir)
  # print(args.data_dir, args.model)
  # sys.exit()
  path_eval_queries = os.path.join(args.data_dir, 'test/queries.txt')
  path_eval_articles = os.path.join(args.data_dir, 'test/articles.txt')

  print("Loading queries and articles text...")
  queries = load_by_line(path_eval_queries)#, 10)
  articles = load_by_line(path_eval_articles)#, 10)
  # articles = load_by_line(os.path.join(args.data_dir, "test/articles.txt")) + \
            #  load_by_line(os.path.join(args.data_dir, "dev/articles.txt"))
  print("Done.\n")

  print("Loading query and article embeddings...")
  t1 = time.time()
  if args.method == "embed":
    from load_embed import Embedding
    embed = Embedding(save_path=args.model, data_path=args.data_dir)
    articles_embed = np.array([embed.get_article_embed(a.split(), norm=False) for a in articles])
    queries_embed = np.array([embed.get_query_embed(q.split(), norm=False) for q in queries])
  elif args.method == "sif":
    from load_embed import Embedding
    embed = Embedding(save_path=args.model, data_path=args.data_dir, mode="sif")
    articles_embed = np.array([embed.get_article_embed(a.split(), norm=False) for a in articles])
    queries_embed = np.array([embed.get_query_embed(q.split(), norm=False) for q in queries])
  elif args.method == "tfidf":
    with open(os.path.join(args.data_dir, "tfidfvec_test2.pkl"), "rb") as f:
      embed = pickle.load(f)
    articles_embed = embed.transform(articles)
    queries_embed = embed.transform(queries)
  elif args.method == "lm":
    from sentence_transformers import SentenceTransformer
    # embed = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    embed = SentenceTransformer('msmarco-distilbert-base-v2')
    articles_embed = np.array(embed.encode(articles, batch_size=100, show_progress_bar=True))
    queries_embed = np.array(embed.encode(queries, batch_size=100, show_progress_bar=True))
  elif args.method == "use":
    import tensorflow_hub as hub
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(module_url)
    import tensorflow as tf
    config = tf.ConfigProto()
    config.graph_options.rewrite_options.shape_optimization = 2
    session = tf.Session(config=config)
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    articles_embed = session.run(embed(articles))
    queries_embed = session.run(embed(queries))
    session.close()
  
  
  if args.method != "tfidf":
    print("Normalizing before dot product...")
    articles_embed /= np.linalg.norm(articles_embed, axis=-1, keepdims=True)
    queries_embed /= np.linalg.norm(queries_embed, axis=-1, keepdims=True)
  print("Done in {}\n".format(datetime.timedelta(seconds=time.time() - t1)))

  t1 = time.time()
  if args.mode in ["rank", "ranktf"]:
    adist = np.dot(queries_embed, articles_embed.T)
  # print(adist)

  if args.method == "tfidf":
    adist = adist.toarray()
  
  if args.mode in ["ranktf", "rerank", "reranktf"]:
    with open(os.path.join(args.data_dir,"tfidfvec_test2.pkl"), "rb") as f:
      tfidf = pickle.load(f)
    articles_embed_tf = tfidf.transform(articles)
    queries_embed_tf = tfidf.transform(queries)
    print("Done in {}\n".format(datetime.timedelta(seconds=time.time() - t1)))
    adist_tf = np.dot(queries_embed_tf, articles_embed_tf.T).toarray()
  if args.mode == "ranktf":  
    adist += adist_tf
  elif args.mode in ["rerank", "reranktf"]:
    adist = adist_tf
  print(adist.shape)
  # sys.exit()

  rr = np.zeros((len(queries),))
  precision1 = np.zeros((len(queries),))
  precision3 = np.zeros((len(queries),))
  precision10 = np.zeros((len(queries),))
  pred_list = []
  for i, d in enumerate(adist):
    ids = d.argsort()[::-1]
    pred_list.append(ids[:10])
    
    if args.mode in ["rank", "ranktf"]:
      precision1[i] = float(ids[0] == i)
      precision3[i] = float(i in ids[:3])
      precision10[i] = float(i in ids[:10])
      r = np.where(ids==i)
      rr[i] = 1 / (r[0][0]+1)
  
    elif args.mode in ["rerank", "reranktf"]:
      ranked_embed = articles_embed[ids[:1000]]
      re_adist = np.dot(queries_embed[i], ranked_embed.T)
      if args.mode == "reranktf":
        re_adist += adist_tf[i][ids[:1000]]
      re_ids = re_adist.argsort()[::-1]
      precision1[i] = float(ids[re_ids[0]] == i)
      precision3[i] = float(i in ids[re_ids[:3]])
      precision10[i] = float(i in ids[re_ids[:10]])
      try:
        r = np.where(ids[re_ids] == i)
        rr[i] = 1 / (r[0][0]+1)
      except:
        rr[i] = 1 / len(queries)
    
  print("Mean Reciprocal Rank:", np.mean(rr) * 100)
  print("Precision@1:", np.mean(precision1) * 100)
  print("Precision@3:", np.mean(precision3) * 100)
  print("Precision@10:", np.mean(precision10) * 100)
  print("Ranking time: {}".format(datetime.timedelta(seconds=time.time() - t1)))
  
  # with open('experiments/signal-news-tf-msmarco-bert-pred.txt', 'w') as f:
  #   for i in pred_list:
  #     f.write(str(i) + '\n')

  sys.exit(0)

  while True:
    try:
      text = input("\nEnter query index\t")
      try:
        index = int(text)
        example = queries[index]
        query_embed = queries_embed[index]
        query_index = True
      except:
        example = text
        query_embed = embed.get_query_embed(example.split(), norm=True)
        query_index = False
      
      print("Query:", example)
      if query_index:
        print("Actual Article:", articles[index][:500])
      print("\nAritcles:")
      ids, dots = rank(query_embed, articles_embed, num=5)
      for r in range(len(ids)):
        print("Rank %d: Index %d, prob = %.2f" %(r+1, ids[r], sigmoid(dots[r])), articles[ids[r]][:500], "\n")

    except KeyboardInterrupt:
      print()
      break
    

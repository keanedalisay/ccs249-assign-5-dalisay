import re

from prettytable import PrettyTable

import wikipedia
import matplotlib.pyplot as plt
import gensim
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.test.gensim_fixt import setup_module
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def tsne_plot(model):
  labels = []
  count = 0
  max_count = 25
  X = np.zeros(shape=(max_count,len(model.wv["overnight"])))

  for term in model.wv.index_to_key:
    X[count] = model.wv[term]
    labels.append(term)
    count+= 1
    if count >= max_count: break

  # It is recommended to use PCA first to reduce to ~25 dimensions
  pca = PCA(n_components=25)
  X_50 = pca.fit_transform(X)

  # Using TSNE to further reduce to 2 dimensions
  model_tsne = TSNE(n_components=2, random_state=0)
  Y = model_tsne.fit_transform(X_50)

  # Show the scatter plot
  plt.scatter(Y[:,0], Y[:,1], 20)

  # Add labels
  for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy = (x,y), xytext = (0, 0), textcoords = "offset points", size = 10)
  
  # plt.show()
  plt.savefig("term_similarity.png")

def main():
  setup_module()

  topics = ['pandesal', 'puto (food)', 'sapin-sapin', 'bibingka', 'kutsinta']
  documents = [wikipedia.page(topic) for topic in topics]
  STOP_WORDS = stopwords.words()

  wiki_texts = []
  for document in documents:
    wiki_text = re.sub(r'[\?\,\(\)\.\[\]\'\`\-\"\/;=_:]', ' ', document.content[:8004].lower()) # This is approximately 1159 words, including new lines.
    wiki_texts.append(wiki_text)
  
  wiki_texts_tokens = [word_tokenize(text, language='english') for text in wiki_texts]
  wiki_texts_tokens = [[word for word in text if word not in STOP_WORDS] for text in wiki_texts_tokens]

  # model = gensim.models.Word2Vec(wiki_texts_tokens, vector_size=100, window=5, min_count=1, workers=4)
  # model.save("kakanin.model")

  TEST_TERM = "overnight"
  model = gensim.models.Word2Vec.load("kakanin.model")
  similar_terms = model.wv.most_similar(TEST_TERM, topn=5)

  similarity_matrix = PrettyTable()
  similarity_matrix.field_names = ["Term", "Cosine Similarity"]

  for term, similarity in similar_terms:
    similarity_matrix.add_row([term, similarity])

  print(f"\nCosine Similarity between \"{TEST_TERM}\" and other terms:")
  print(similarity_matrix)

  tsne_plot(model)

if __name__ == "__main__":
  main()
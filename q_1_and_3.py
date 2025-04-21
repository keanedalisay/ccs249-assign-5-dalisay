import re, math

from collections import Counter
from math import log
from prettytable import PrettyTable

import wikipedia

from nltk.tokenize import word_tokenize

def compute_tf(tokens, vocab):
    count = Counter(tokens)
    total_terms = len(tokens)
    return { term: count[term] / total_terms for term in vocab }

def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        # Count the number of documents containing the term
        df = sum(term in doc for doc in tokenized_docs)
        # Compute IDF using the formula: idf(t) = lo(N / df(t))
        idf_dict[term] = log(N / (df or 1))
    return idf_dict

def compute_tfidf(tf_vector, idf, vocab):
    # Compute the TF-IDF score for each term in the vocabulary
    # using the formula: tf-idf(t, d) = tf(t, d) * idf(t)
    return { term: tf_vector[term] * idf[term] for term in vocab}

def cosine_similarity(vec1, vec2, vocab):
    # Get the dot product of the two vectors
    dot_product = sum(vec1[term] * vec2[term] for term in vocab)
    vec1Length = math.sqrt(sum(vec1[term]**2 for term in vocab))
    vec2Length = math.sqrt(sum(vec2[term]**2 for term in vocab))

    if vec1Length == 0 or vec2Length == 0:
        return 0.0

    return dot_product / (vec1Length * vec2Length)

def main():
  topics = ['pandesal', 'puto (food)', 'sapin-sapin', 'bibingka', 'kutsinta']
  documents = [wikipedia.page(topic) for topic in topics]

  wiki_texts = []
  for document in documents:
    wiki_text = re.sub(r'[\?\,\(\)\.\[\]\'\`\-\"\/;=_:]', ' ', document.content[:8004].lower()) # This is approximately 1159 words, including new lines.
    wiki_texts.append(wiki_text)
  
  wiki_texts_tokens = [word_tokenize(text, language='english') for text in wiki_texts]

  vocabulary = set(token for wiki_text_tokens in wiki_texts_tokens for token in wiki_text_tokens)

  tf_vectors =  [compute_tf(wiki_text_tokens, vocabulary) for wiki_text_tokens in wiki_texts_tokens]

  # tf_matrix = PrettyTable()
  # tf_matrix.field_names = ["Term"] + [f"\"{topics[i]}\"" for i in range(len(topics))]
  # for term in vocabulary:
  #     tf_matrix.add_row([term] + [tf_vector[term] for tf_vector in tf_vectors])
  # print(tf_matrix)

  idf_vectors = compute_idf(wiki_texts_tokens, vocabulary)

  tfidf_vectors = [ compute_tfidf(tf, idf_vectors, vocabulary) for tf in tf_vectors ]

  # tfidf_matrix = PrettyTable()
  # tfidf_matrix.field_names = ["Term"] + [f"\"{topics[i]}\"" for i in range(len(topics))]
  # for term in vocabulary:
  #     tfidf_matrix.add_row([term] + [tfidf_vector[term] for tfidf_vector in tfidf_vectors])
  # print(tfidf_matrix)

  # Compute the Cosine Similarity between the first two documents
  similarity = cosine_similarity(tfidf_vectors[2], tfidf_vectors[4], vocabulary)
  print(f"\nCosine Similarity between \"{topics[2]}\" and \"{topics[4]}\":")
  print(similarity)
    

if __name__ == "__main__":
  main()
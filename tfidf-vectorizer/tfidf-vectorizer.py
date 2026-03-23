import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):

    #step 1: Tokenize
    tokenized_docs = [doc.split() for doc in documents]

    #step 2: Vocubulary
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    vocab_index = {word: i for i, word in enumerate(vocab)}

    N = len(documents)

    # step 3: Document Frequency 
    df = Counter()
    for doc in tokenized_docs:
        unique_words = set(doc)
        for word in unique_words:
            df[word] +=  1

    # step 4: IDF
    idf = {}
    for word in vocab:
        idf[word] = math.log(N / df[word])


    # step 5: TF-IDF Matrix
    tfidf_matrix = np.zeros((N, len(vocab)))

    for i, doc in enumerate(tokenized_docs):
        word_count = Counter(doc)
        total_words = len(doc)

        for word in word_count:
            tf = word_count[word] / total_words
            tfidf = tf * idf[word]
            j = vocab_index[word]
            tfidf_matrix[i][j] = tfidf
    return tfidf_matrix, vocab
        
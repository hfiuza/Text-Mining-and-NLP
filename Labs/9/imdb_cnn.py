# CNN for text classification.
# based on https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
# Obtained loss and accuracy:
# - training loss: 0.4680
# - training accuracy: 0.7778
# - validation loss: 0.3207
# - validation accuracy: 0.8659

import os
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb
from collections import Counter

from gensim import utils
from gensim.models.word2vec import Word2Vec

np.random.seed(1337)  # for reproducibility

# set parameters:
max_features = 10000 # only the 'max_features' most frequent words will constitute the dictionary
maxlen = 400 # truncate documents longer than this number of words (among top 'max_features' most common words)
batch_size = 120
word_vector_dim = 300 # dimensionality of the word vectors
nb_filter = 250
filter_length = 3 # region size
hidden_dims = nb_filter # input size of the final fully connected layer
nb_epoch = 1
d_e = 0.2 # dropout for the embedding layer, see http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

print 'Loading data...'
# load the dataset in a format that is ready for use by neural network
# https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print len(X_train), 'train sequences'
print len(X_test), 'test sequences'

# note: it is a 1-based index - the size is much bigger than max_features
# because it provides indexing for all the unique words in the full vocab
words_to_index = imdb.get_word_index()

# invert mapping
index_to_words = dict((v,k) for k, v in words_to_index.iteritems())

# convert dataset (list of lists of integers) into list of lists of words
X_full = X_train.tolist() + X_test.tolist()

all_idxs = [item for sublist in X_full for item in sublist]

word_docs = []
for elt in X_full:
    word_docs.append([index_to_words[idx] for idx in elt])

# ====== start: if using pre-trained embeddings ======

mcount = 5
word_vectors = Word2Vec(size=word_vector_dim, min_count=mcount)

# instead of randomly initializing the word embeddings, we can use Google News ones
# load only the ones corresponding to our vocabulary
word_vectors.build_vocab(word_docs)

#all_words = [item for sublist in word_docs for item in sublist]
#counts = dict(Counter(all_words))

# sanity check (should return True)
max(all_idxs) == max_features - 1 == len(word_vectors.wv.vocab) + 1
# one of the indexes is for unknown words

# sanity check
#len(word_vectors.wv.vocab) == len([k for (k,v) in counts.iteritems() if v>mcount])
# actually, all the words appear at least mcount times
# surely due to the preprocessing that has been done on the reviews

# load Google news word vectors corresponding to the words in our vocab
os.chdir('data/')
word_vectors.intersect_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# len(word_vectors.wv.vocab) is still the same - so all the words in our vocab have an entry in the Google binary file
#out_of_vocab_words = [word for word in word_vectors.wv.vocab if not np.count_nonzero(word_vectors[word])]

# will be passed to the Embedding layer later on
embedding_weights = np.zeros((max_features,word_vector_dim))
for word in word_vectors.wv.vocab.keys():
    idx = words_to_index[word]
    embedding_weights[idx,] = word_vectors[word]

# ====== end: if using pre-trained embeddings ======

print 'Padding sequences...'
# pad each sequence to the same length (length of the longest sequence)
# if maxlen is provided, any sequence longer than maxlen is truncated to maxlen
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print 'X_train shape:', X_train.shape
print 'X_test shape:', X_test.shape

print 'Building model...'

model = Sequential() #https://keras.io/getting-started/sequential-model-guide/

# if using pre-trained embeddings
model.add(Embedding(max_features,word_vector_dim,input_length=maxlen,dropout=d_e, weights=[embedding_weights])) #https://keras.io/layers/embeddings/

# if not using pre-trained embeddings
#model.add(Embedding(max_features,word_vector_dim,input_length=maxlen,dropout=d_e)) #https://keras.io/layers/embeddings/

# add convolution layer, which will learn nb_filter word group filters of size filter_length
# https://keras.io/layers/convolutional/
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        activation='relu',
                        ))

model.add(GlobalMaxPooling1D()) # https://keras.io/layers/pooling/

# optional: add a vanilla hidden layer (affine layer)
#model.add(Dense(hidden_dims,  activation='sigmoid')) # https://keras.io/layers/core/#dense
#model.add(Dropout(0.2))

# we finally project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print 'Training model...'

model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

# not needed here because we use the test set as validation data
#score,acc = model.evaluate(X_test, y_test,batch_size=batch_size)
#print 'Test score:', score
#print 'Test accuracy:', acc

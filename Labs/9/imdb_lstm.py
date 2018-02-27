# trains a LSTM on the IMDB sentiment classification task

# based on https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, SimpleRNN
from keras.datasets import imdb
# requires pydot_ng
#from keras.utils.visualize_util import plot

np.random.seed(1337) # for reproducibility

# tuning parameters
max_features = 20000 # only the 'max_features' most frequent words will constitute the dictionary
maxlen = 80  # truncate documents longer than this number of words (among top 'max_features' most common words)
batch_size = 100
word_vector_dim = 128 # dimension of the embedding vectors
nb_epoch = 1
d_e = 0.2 # dropout for the embedding layer, see http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
d_U = 0.2 # dropout for recurrent connections
d_W = 0.2 # dropout for input gates

print 'Loading data...'
# load the dataset in a format that is ready for use by neural network
# https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print len(X_train), 'train sequences'
print len(X_test), 'test sequences'

print 'Padding sequences...'
# pad each sequence to the same length (length of the longest sequence)
# if maxlen is provided, any sequence longer than maxlen is truncated to maxlen
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print 'X_train shape:', X_train.shape
print 'X_test shape:', X_test.shape

print 'Building model...'

model = Sequential() #https://keras.io/getting-started/sequential-model-guide/
model.add(Embedding(max_features, word_vector_dim, dropout=d_e)) #https://keras.io/layers/embeddings/
model.add(LSTM(word_vector_dim, dropout_W=d_W, dropout_U=d_U)) # https://keras.io/layers/recurrent/#lstm
model.add(Dense(1, activation='sigmoid')) #https://keras.io/layers/core/#dense

# https://keras.io/models/model/#methods
# https://keras.io/optimizers/
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print 'Plotting model...'

plot(model, to_file='model.png', show_shapes=True)

print 'Training model...'

# https://keras.io/getting-started/sequential-model-guide/#training
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

# returns the loss value and performance metrics values for the model in test mode. Computation is done in batches
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print 'Test score:', score
print 'Test accuracy:', acc

# ======= simple RNN =========

model = Sequential() #https://keras.io/getting-started/sequential-model-guide/
model.add(Embedding(max_features, word_vector_dim, dropout=d_e)) #https://keras.io/layers/embeddings/
model.add(SimpleRNN(word_vector_dim, dropout_W=d_W, dropout_U=d_U)) # https://keras.io/layers/recurrent/#lstm
model.add(Dense(1, activation='sigmoid')) #https://keras.io/layers/core/#dense

# https://keras.io/models/model/#methods
# https://keras.io/optimizers/
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# https://keras.io/getting-started/sequential-model-guide/#training
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

# returns the loss value and performance metrics values for the model in test mode. Computation is done in batches
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print 'Test score:', score
print 'Test accuracy:', acc
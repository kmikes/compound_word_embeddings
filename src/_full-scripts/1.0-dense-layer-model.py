import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import torchtext
import os
import pathlib

from numpy import random
from scipy import spatial

import torch.nn as nn
from torch.optim import Adam

dims = 50
drop = 0.1

# Load Data
data = pd.read_csv('data/processed/all_embeddings_forML.csv')

voc = np.unique( data[ ['c1', 'c2', 'cmp'] ].values.reshape(-1) )
# print( voc.shape )

# Shuffle the data
random.seed(1)
np.random.seed(1)

df = data.copy()
# print(df.head())
df = df.sample(frac=1).reset_index(drop=True)
# print(df.head())

# Extract a training & validation split
c1 = list(df['c1'])
c2 = list(df['c2'])
compounds = list(df['cmp'])
rows = len(compounds)

print("Loading glove embeddings..")
word_index = dict(zip(voc, range(len(voc))))
vocab = torchtext.vocab.GloVe(name='6B',dim=dims)
vocab_words = set(vocab.itos)
hits, misses = 0,0
for w in word_index.keys():
    if w in vocab_words:
        hits=hits+1
    else:
        misses = misses+1

print("Word count: ", len(word_index))
print("Hits/misses: ", hits, '/', misses)

#vocab = glove.vocab(word_index)
cmp_embeddings = np.array( df.iloc[:int((0.8*(len(data['c1'])))), 104:154], dtype='float32' )

x_c1 = [vocab[w] for w in c1]
x_c2 = [vocab[w] for w in c2]

xx = [ tf.concat([a,b], axis=0) for a,b in zip(x_c1, x_c2) ]

X = tf.stack( xx )
y = tf.stack([vocab[w] for w in compounds])

rows_train = int(0.8 * rows)
rows_test = rows - rows_train

## X_train = 80% of rows, shape=(None,2,dims) tensor
## y_train = shape=(None,dims) tensor
X_train, X_test = tf.split(X, num_or_size_splits=[rows_train, rows_test])
y_train, y_test = tf.split(y, num_or_size_splits=[rows_train, rows_test])

# Build Model
from tensorflow.keras import layers

input_shape = (2*dims)
# Each X is a tensor with shape=(None, 2*dims)
# Timestep is probably 1

input = keras.Input(shape=input_shape, dtype="float64")

x = layers.Dense(128, activation='relu')(input)
x = layers.Dense(256, activation='linear')(x)
#x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(drop)(x)

x = layers.Dense(512, activation='relu')(x)
#x = layers.Dense(512, activation='linear')(x)
x = layers.Dropout(drop)(x)

x = layers.Dense(1024, activation='linear')(x)
x = layers.Dense(1024, activation='linear')(x)
x = layers.Dropout(drop)(x)

x = layers.Dense(1024, activation='linear')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(drop)(x)
x = layers.Dense(dims, activation='linear')(x)

# I'm almost certain that the values in glove are always between -5 and 5
#layer = layers.experimental.preprocessing.Normalization( mean=0, variance=25 ) #Supposed to change the numbers in the predictions to better match the embeddings
#x = layer(x)

model = keras.Model(input, x)
model.summary()

# Train Model
model.compile(
    loss="cosine_similarity", optimizer="Adam", metrics=["acc"]
)
model.fit(X_train, y_train, batch_size=16, epochs=20, validation_split=0.25)

result = model.evaluate(X_test, y_test)
print("test loss, test acc:", result)

samples = 3
print("Generate predictions for ", samples, " samples")
predictions = model.predict(X_test[:samples])
print("predictions shape:", predictions.shape)

def find_closest_embeddings(vocab, embedding):
    return sorted(vocab.itos[:1000], key=lambda word: spatial.distance.euclidean(vocab[word], embedding))

print('True Embeddings')
for i in range(samples):
    print( df.loc[rows_train+i,['c1', 'c2', 'cmp']], find_closest_embeddings(vocab, y_test[i] )[:5] )
print('')

print('Predicted Embeddings')
for i in range(samples):
    print( df.loc[rows_train+i,['c1', 'c2', 'cmp']], find_closest_embeddings(vocab, predictions[i])[:5] )
print('')

#'''
print('')
for i in range(samples):
    print('True Embedding:')
    print( y_test[i] )
    print('')
    print('Predicted Embedding:')
    print( predictions[i] )
    print('')
# '''

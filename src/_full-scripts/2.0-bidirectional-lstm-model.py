# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""

from __future__ import division, print_function, absolute_import

import torchtext
import pandas as pd
import numpy as np
import tensorflow as tf

from numpy import random
from scipy import spatial

from tensorflow import keras

# Load Data
data = pd.read_csv('../../data/processed/all_embeddings_forML.csv')

voc = np.unique( data[ ['c1', 'c2', 'cmp'] ].values.reshape(-1) )
# print( voc.shape )

dims = 50

# Shuffle the data
random.seed(1)
np.random.seed(1)

df = data.copy()
# print(df.head())
df = df.sample(frac=1).reset_index(drop=True)
# print(df.head())

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

#SETUP VISIBLE TESTING
compounds = list(df['cmp'])
rows = len(compounds)
rows_train = int(0.8 * rows)
rows_test = rows - rows_train

cmp_embeddings = np.array( df.iloc[:int((0.8*(len(data['c1'])))), 104:154], dtype='float32' )

# trainX - np array of float32, 80% of the data
trainX =  df.iloc[:int((0.8*(len(data['c1'])))), 4:104]
trainY =  df.iloc[:int((0.8*(len(data['c1'])))), 104:154]

testX = df.iloc[int((0.8*(len(data['c1'])))):, 4:104]
testY =  df.iloc[int((0.8*(len(data['c1'])))):, 104:154]

'''
print('')
print( 'trainX', trainX.shape )
print( 'trainY', trainY.shape)
print( 'testX', testX.shape )
print( 'testY', testY.shape )
print('')
'''


# Network building
from keras import layers

model = keras.Sequential(
    [
        layers.Reshape( (4,25), input_shape=(100,)),
        layers.SimpleRNN(128),
        #layers.Dense(512, activation='linear'),
        #layers.Dense(1024, activation='linear'),
        #layers.Dense(128, activation='linear'),
        layers.Dense(50, activation='linear')
    ]
)

'''
foo = testX
bar = model(foo)
print("Number of weights after calling the model:", len(model.weights))  # 5

model.summary()
'''

# Training
model.compile(loss="mean_squared_error", optimizer="Adam", metrics=["acc"])
model.fit( trainX, trainY, batch_size=16, epochs=40, validation_split=0.25)

model.summary()

result = model.evaluate(testX, testY)
print("test loss, test acc:", result)

#Visible Testing
samples = 3
print("Generate predictions for ", samples, " samples")
predictions = model.predict(testX[:samples])
print("predictions shape:", predictions.shape)

def find_closest_embeddings(vocab, embedding):
    return sorted(vocab.itos[:1000],
                      key=lambda word: spatial.distance.euclidean(vocab[word], embedding))

print('')
for i in range(samples):
    print( df.loc[rows_test+i,['c1', 'c2', 'cmp']], find_closest_embeddings(vocab, predictions[i])[:5] )
print('')

# '''
print('')
for i in range(samples):
    print('True Embedding:')
    print( testY[i] )
    print('')
    print('Predicted Embedding:')
    print( predictions[i] )
    print('')
# '''

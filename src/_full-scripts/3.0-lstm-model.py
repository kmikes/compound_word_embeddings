import pandas as pd
import numpy as np

import tflearn

from scipy import spatial
from numpy import random


# Load Data
data = pd.read_csv('../../data/processed/all_embeddings_forML.csv')

voc = np.unique(data[['c1', 'c2', 'cmp']].values.reshape(-1))

random.seed(1)
np.random.seed(1)

df = data.copy()
#print(df.head())
df = df.sample(frac=1).reset_index(drop=True)
#print(df.head())


# EMBEDDINGS DICT
embeddings_dict = {}

with open("../../data/external/glove.6B.50d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


# SETUP X and Y
compounds = list(df['cmp'])
rows = len(compounds)
rows_train = int(0.8 * rows)
rows_test = rows - rows_train


# trainX - np array of float32, 80% of the data
trainX = np.array(df.iloc[:rows_train, 4:104], dtype='float32')
trainY = np.array(df.iloc[:rows_train, 104:154], dtype='float32')

testX = np.array(df.iloc[rows_train:, 4:104], dtype='float32')
testY = np.array(df.iloc[rows_train:, 104:154], dtype='float32')


# Building Network
net = tflearn.layers.core.input_data( shape=[None, 100] )
# net = tflearn.layers.embedding_ops.embedding(net, rows, 128)

net = tflearn.layers.reshape(net, [-1, 2, 50])
net = tflearn.layers.recurrent.lstm(net, 128, return_seq=True)
net = tflearn.layers.core.dropout (net, 0.8)

net = tflearn.layers.recurrent.lstm(net, 128, return_seq=False)
#net = tflearn.layers.core.dropout(net, 0.8)

net = tflearn.layers.core.fully_connected(net, 50, activation='linear')
net = tflearn.layers.estimator.regression(net, optimizer='adam', learning_rate=0.001, loss='mean_square')


# Training Network
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=40, validation_set=0.25, show_metric=True, batch_size=4)


# Results
result = model.evaluate(testX, testY)
print("test acc:", result)


# Visible Testing
samples = 3
print("Generate predictions for ", samples, " samples")
predictions = model.predict(testX[:samples])
print("predictions shape:", predictions.shape)
print('')
print('')

'''
print('TRUE EMBEDDING')
for i in range(samples):
    print(df.loc[rows_train + i, ['c1', 'c2', 'cmp']], find_closest_embeddings( testY[i] )[:5])
print('')
'''

print('PREDICTED EMBEDDING')
for i in range(samples):
    print(df.loc[rows_train + i, ['c1', 'c2', 'cmp']], find_closest_embeddings( predictions[i] )[:5])
print('')


for i in range(samples):
    print('True Embedding:')
    print( testY[i] )
    print('')
    print('Predicted Embedding:')
    print( predictions[i] )
    print('')


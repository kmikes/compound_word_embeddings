#!/bin/sh
# expect to be run from the top of the repository

# make data subdirectories
mkdir -p data/external data/processed

# download glove embeddings
curl -o data/external/glove.6B.zip -C - http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip -u data/external/glove.6B.zip glove.6B.50d.txt
mv glove.6B.50d.txt data/external

curl -o data/external/ladec.csv -C - https://era.library.ualberta.ca/items/dc3b9033-14d0-48d7-b6fa-6398a30e61e4/download/830937da-a00b-4735-8cf2-3c67d5cc6d50

shasum -c data/SHASUMS.txt

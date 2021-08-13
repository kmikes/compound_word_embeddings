compound_word_embeddings
==============================

About This Project
------------
This project focuses on the creation of Neural Networks that attempt to create an embedding for a compound word based on the embeddings of the two constituent words. This project uses compound word data from the <a target="_blank" href="https://era.library.ualberta.ca/items/dc3b9033-14d0-48d7-b6fa-6398a30e61e4">LaDEC</a> dataset and word embeddings from <a target="_blank" href="https://nlp.stanford.edu/projects/glove/">GloVe: Global Vectors for Word Representation</a>.

The original papers for both of these resources are listed:
- <a target="_blank" href="https://link.springer.com/article/10.3758/s13428-019-01282-6">LaDEC</a>
- <a target="_blank" href="https://nlp.stanford.edu/pubs/glove.pdf">GloVe</a>


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    |   ├── _full-scripts  <- Contains scripts that build, run, and evaluate models using the data
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py


--------


What We've Done So Far
------------
We have created 3 models that take in the constituent embeddings as input and the compound embeddings as desired output.
1. The first model uses <a target="_blank" href="https://keras.io">keras</a> and uses dense layers.
2. The second model uses <a target="_blank" href="http://tflearn.org">tflearn</a> and lstm layers. This model has 4 bidirectional lstm layers.
3. The third model uses <a target="_blank" href="http://tflearn.org">tflearn</a> as well. It has regular lstm layers.

We believe that more complex architecture is required to tackle this problem since none of these models have been able to achieve accuracy above 25%.


Tips for LaDEC
------------
It is important to note two things about LaDEC:
1. Not all of the compound words included in the dataset have embeddings in GloVe, so these words must be filtered out before the data is run through any Neural Networks.
2. Not all of the compounds in LaDEC are 'correct parses' of their constituent words. LaDEC, for example, includes the word: 'wholesale' twice, once with constituents 'whole' and 'sale' and once with constituents 'wholes' and 'ales'. To avoid this, all incorrect parses should also be filtered out.


Tips for GloVe
------------
We reccomend starting by downloading the 50d GloVe vectors, and only moving to larger vectors once the 50d vectors are easy to use and manipulate.

Once downloaded there are two easy ways to process the vectors:
1. Manual Vector Extraction: Create a dictionary and process words/strings as keys and their embeddings as values. This approach may be helped by eliminating non-words before they are added to the dictionary.
2. TorchText Extraction: Use <a target="_blank" href="https://torchtext.readthedocs.io/en/latest/">torchtext</a> to create a vocab to hold the words and their embeddings. Using torchtext allows you to filter the dataset by word frequency when finding nearest embeddings.


Next Steps & Research Ideas
------------


Reaching Out
------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

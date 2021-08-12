compound_word_embeddings
==============================

About This Project
------------
This project focuses on the creation of Neural Networks that attempt to create an embedding for a compound word based on the embeddings of the two constituents. This project uses compound word data from the <a target="_blank" href="https://era.library.ualberta.ca/items/dc3b9033-14d0-48d7-b6fa-6398a30e61e4">LaDEC</a> dataset and word embeddings from <a target="_blank" href="https://nlp.stanford.edu/projects/glove/">GloVe: Global Vectors for Word Representation</a>.

The original papers for both of these resources are listed:
- <a target="_blank" href="https://link.springer.com/article/10.3758/s13428-019-01282-6">LaDEC</a>
- <a target="_blank" href="https://nlp.stanford.edu/pubs/glove.pdf">GloVe</a>


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
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
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


What We've Done So Far
------------
We have created 3 models that take in the constituent embeddings as input and the compound embeddings as desired output.
1. The first model uses <a target="_blank" href="https://keras.io">keras</a> and uses dense layers.
2. The second model uses <a target="_blank" href="http://tflearn.org">tflearn</a> and lstm layers. This model has 4 bidirectional lstm layers.
3. The third model uses <a target="_blank" href="http://tflearn.org">tflearn</a> as well. It has regular lstm layers.

We believe that more complex architecture is required to tackle this problem since all of these models have not been able to achieve accuracy above 25%.

Next Steps & Research Ideas
------------

Reaching Out
------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

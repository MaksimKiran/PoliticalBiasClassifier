import torch
from sklearn.naive_bayes import  ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

from preprocess import get_cached_embeddings

class SBERTTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cache_path = f"../data/embeddings_{len(X)}.npy"
        return get_cached_embeddings(X, cache_path, self.model)

def train_model(X_train, y_train, model_type="logreg", SBERT = False, bert_name = "bert-base-nli-mean-tokens"):
    """
    Trains a model based on the model_type argument and returns it as a Pipeline object.

    :param X_train: features in train set
    :param y_train: labels in train set
    :param model_type: model used
    :param SBERT: boolean to indicate whether to use sentence embeddings through SBERT
    :param bert_name: name of BERT model
    :return: pipeline object trained on the data provided through classifier specified by model_type
    """

    if model_type == "logreg":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    elif model_type == "svm":
        classifier = LinearSVC(class_weight="balanced")
    elif model_type == "nb":
        classifier = ComplementNB()
    else:
        raise ValueError("model_type must be 'logreg' or 'svm'")

    if SBERT:
        pipeline = Pipeline([
            ("SBERTTransformer", SBERTTransformer(bert_name)),
            ("clf", classifier)
        ])
    else:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2),
                min_df=5
            )),
            ("clf", classifier)
        ])

    pipeline.fit(X_train, y_train)
    return pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

def train_model(X_train, y_train, model_type="logreg"):
    """
    Trains a model based on the model_type argument and returns it as a Pipeline object.

    :param X_train: features in train set
    :param y_train: labels in train set
    :param model_type: model used
    :return: pipeline object trained on the data provided through classifier specified by model_type
    """

    if model_type == "logreg":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    elif model_type == "svm":
        classifier = LinearSVC()
    else:
        raise ValueError("model_type must be 'logreg' or 'svm'")

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
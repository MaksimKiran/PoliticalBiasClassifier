from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

from preprocess import load_and_split_data
from train import train_model
from evaluate import evaluate_model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_split_data()
    pipeline = train_model(X_train, y_train)
    evaluate_model(pipeline, X_test, y_test)





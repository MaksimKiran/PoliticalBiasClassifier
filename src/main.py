from preprocess import load_and_split_data
from train import train_model
from evaluate import evaluate_model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_split_data()
    pipeline = train_model(X_train, y_train, model_type="logreg", SBERT=False,bert_name="all-MiniLM-L6-v2")
    evaluate_model(pipeline, X_test, y_test)
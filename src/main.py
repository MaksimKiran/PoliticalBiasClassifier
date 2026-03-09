from preprocess import load_and_split_data, get_tiny_subset
from train import train_model
from evaluate import evaluate_model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_split_data()
    pipeline = train_model(X_train, y_train, model_type="svm", SBERT=True, bert_name="all-MiniLM-L6-v2")
    evaluate_model(pipeline, X_test, y_test)

    # Proof of overfit
    # X_train, X_test, y_train, y_test = load_and_split_data()
    # X_tiny, y_tiny = get_tiny_subset(X_train, y_train, n_per_class=10)
    # pipeline = train_model(X_tiny, y_tiny, model_type="svm")
    # evaluate_model(pipeline, X_tiny, y_tiny)
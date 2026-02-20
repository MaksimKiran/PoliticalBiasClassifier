from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Macro:", f1_score(y_test, y_pred, average="macro"))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

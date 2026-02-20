from sklearn.model_selection import train_test_split
import pandas as pd

def load_and_split_data(path="../data/allsides_balanced_news_headlines-texts.csv", ftest_size=0.3, frandom_state=11):
    """
    Loads from file through path and splits it into train and test sets.

    :param path: file path
    :param ftest_size: size of the test set, value between 0 and 1.
    :param frandom_state: random state of the split.
    :return: 4 arrays containing train and test for X and train and test for y in that order.

    """

    df = pd.read_csv(path)
    # Only text and bias rating matter
    df = df[["text", "bias_rating"]]
    df = df.dropna()
    # Readability, optional
    # df.to_csv("../data/cleaned_data.csv", index=False)

    X = df["text"]
    y = df["bias_rating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=frandom_state,
                                                        test_size=ftest_size,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


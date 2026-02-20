from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("data/allsides_balanced_news_headlines-texts.csv")
    # Only text and bias rating matter
    df_dropped = df.iloc[:,-2:]
    df_dropped = df_dropped.dropna()
    df_dropped.to_csv("data/cleaned_data.csv", index=False)



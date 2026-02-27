# Political Bias Classifier using Logistic Regression 


### Abstract 
With the ever-increasing intensity and chaos of the political landscape globally, better methods are required to approximate
the bias of news articles with regard to their political standing. It may no longer be sufficient to label a source's bias purely
based on their track record, as journalistic language increasingly incorporates nuanced phrases, buzzwords and framing that can sway
the narrative of a given story. The aim of this project is to investigate whether political bias 
can be detected from news article text, and to what extent outlet identity drives 
bias classification on a real-world dataset.

### Technical
This project uses the `allsides_balanced_news_headlines-texts.csv` dataset to predict 
political bias ratings (left, center, right). Raw data can be found at this public 
GitHub repository [here.](https://github.com/irgroup/Qbias)

The pipeline consists of three modules:
preprocess.py` loads the CSV, extracts relevant features, performs stratified train/test split.
`train.py` trains a pipeline using a `ColumnTransformer` to handle text (TF-IDF) and 
categorical features (OneHotEncoding) separately, with support for Naive Bayes, 
Logistic Regression, and LinearSVC.
`evaluate.py` reports accuracy, Macro F1, per-class metrics, and confusion matrix.

### Results
| Experiment | Features | Accuracy | Macro F1 |
|---|---|---|---|
| Text only | article body | 0.50 | 0.45 |
| Full features | text + source | 0.99 | 0.99 |

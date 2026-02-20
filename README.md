# Political Bias Classifier using Logistic Regression 


### Abstract 
With the ever-increasing intensity and chaos of the political landscape globally, better methods are required to approximate
the bias of news articles with regard to their political standing. It may no longer be sufficient to label a source's bias purely
based on their track record, as journalistic language increasingly incorporates nuanced phrases, buzzwords and framing that can sway
the narrative of a given story. The aim of this model is to provide a baseline litmus test for the existence (or non-existence) of
bias present in a given text entry.

### Technical
This project uses the allsides_balanced_news_headlines-texts.csv file to predict the political bias of a text entry.
Raw data can be found at this public GitHub repository [here.](https://github.com/irgroup/Qbias)
For now the model uses only the text entry itself for training (this is by design, other features are to be considered as well).
The preprocess script reads the csv file and extract the necessary attributes, then perfroms the split. The train script returns a Pipeline object
which uses one of three models: GaussianNB, Logistic Regression and SVC. Then the evaluate script performs the model evaluation. Metrics include confusion matrix,
Macro F1, as well as basic accuracy and precision metrics.
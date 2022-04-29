
from asreview.models.feature_extraction.base import BaseFeatureExtraction
import pandas as pd                 #For data science purposes
import re                           #For performing regex
import torch                        #For running models with cude
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class Enron(BaseFeatureExtraction):
    """Naive Bayes classifier

    The Naive Bayes classifier with the default SKLearn parameters.
    """

    name = "enron"
    label = "enron_custom_feature_extraction"

    def __init__(self, *args, **kwargs):

        super(Enron, self).__init__(*args, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained("pdelobelle/robbert-v2-dutch-base" )
        self._model.eval()
        self._tokenizernlp = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
        self.sentiment_analysis = pipeline("sentiment-analysis", model=self._model, tokenizer=self._tokenizernlp,
                                      max_length=512,
                                      truncation=True, device=0)
    def transform(self, texts):
        X = texts.apply(lambda x: self.sentiment_analysis(x))

        return X





    def generatesentimentvalues(self,text):
        sentiment_result = self.sentiment_analysis(text)
        if sentiment_result[0]['label'] == 'NEGATIVE':
            result = 0 - sentiment_result[0]['score']
        else:
            result = sentiment_result[0]['score']
        return result


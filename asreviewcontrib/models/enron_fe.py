
from asreview.models.feature_extraction.base import BaseFeatureExtraction
import pandas as pd                 #For data science purposes
import re                           #For performing regex
import torch                        #For running models with cude
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class Enron(BaseFeatureExtraction):
    """Naive Bayes classifier

    The Naive Bayes classifier with the default SKLearn parameters.
    """

    name = "enron"
    label = "enron_custom_feature_extraction"

    def __init__(self):

        super(Enron, self).__init__()
        self._model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english" )
        self._model.eval()
        self._tokenizernlp = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    def transform(self, texts):
        X = self._model.transform(texts).tocsr()
        return X


from asreview.models.feature_extraction.base import BaseFeatureExtraction
import numpy as np

import pandas as pd                 #For data science purposes
import re                           #For performing regex
import torch                        #For running models with cude
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class Enron(BaseFeatureExtraction):
    """Custom feature extraction

    Feature extraction that generates features based on sentiment values and named entity recogntion among other things.
    """

    name = "enron"
    label = "Enron feature extraction"

    def __init__(self, *args, **kwargs):
        self._model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self._tokenizernlp = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        super(Enron, self).__init__(*args, **kwargs)
    #Todo refactor this so that no for loop is used
    def transform(self, texts):
        resultsentiment = np.empty([0])
        resulttextlen = np.empty([0])
        resultspecificwords = np.empty([0])
        for text in texts:
            resultsentiment = np.append(resultsentiment, self.generatesentimentvalues(text))
            resulttextlen = np.append(resulttextlen, self.gettextlength(text))
            resultspecificwords = np.append(resultspecificwords, self.specific_words_check(text))
        resultsentiment = resultsentiment.reshape(-1, 1)
        resulttextlen = resulttextlen.reshape(-1,1)
        resultspecificwords = resultspecificwords.reshape(-1,1)
        result = np.hstack((resultsentiment, resulttextlen, resultspecificwords))

        return result

    def generatesentimentvalues(self, text):
        sentiment_analysis = pipeline("sentiment-analysis", model=self._model, tokenizer=self._tokenizernlp,
                                      max_length=512,
                                      truncation=True, device=0)
        sentiment_result = sentiment_analysis(text)
        if sentiment_result[0]['label'] == 'NEGATIVE':
            result = 0 - sentiment_result[0]['score']
        else:
            result = sentiment_result[0]['score']
        return result

    def gettextlength(self, text):
        return len(text)

    def specific_words_check(self, text):
        ''' Function that searches for specific words and sums the total occurences
        By using regex, this function looks for the words Office, Policy, CAISO, Sales and Ligitiation and counts the amount of times it finds these words and adds it all up
        '''
        amount_of_words = len(re.findall(
            r'(\b[Oo]+ffice\b|\b(?<![@])[Ee]+nron\b|\b[Pp]+olicy\b|\bCAISO\b|\b[Ss]+ales\b|\b[Ll]itigation\b)', text))
        if amount_of_words:
            return (amount_of_words)
        else:
            return 0




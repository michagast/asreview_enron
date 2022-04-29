
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

        super(Enron, self).__init__(*args, **kwargs)
    #Todo refactor this so that no for loop is used
    def transform(self, texts):
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        tokenizernlp = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizernlp,
                                           max_length=512,
                                           truncation=True, device=0)
        result = np.empty([0])
        for text in texts:
            np.append(result, sentiment_analysis(text))
        return result








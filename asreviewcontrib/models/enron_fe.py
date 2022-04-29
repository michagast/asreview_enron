
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

    def transform(self, texts):
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        tokenizernlp = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizernlp,
                                           max_length=512,
                                           truncation=True, device=0)
        apply_sentiment_analysis = lambda x: sentiment_analysis(x)

        return apply_sentiment_analysis(texts)








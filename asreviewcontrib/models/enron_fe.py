
from asreview.models.classifiers.base import BaseFeatureExtraction

class Enron(BaseFeatureExtraction):
    """Naive Bayes classifier

    The Naive Bayes classifier with the default SKLearn parameters.
    """

    name = "enron"
    label = "enron_custom_feature_extraction"

    def __init__(self):

        super(Enron, self).__init__()

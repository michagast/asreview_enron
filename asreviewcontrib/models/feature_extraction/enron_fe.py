

class Enron(BaseTrainClassifier):
    """Naive Bayes classifier

    The Naive Bayes classifier with the default SKLearn parameters.
    """

    name = "enron"
    label = "enron_custom_feature_extraction"

    def __init__(self):

        super(Enron, self).__init__()
        self._model = MultinomialNB()

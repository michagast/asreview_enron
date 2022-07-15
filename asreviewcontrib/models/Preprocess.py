import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from num2words import num2words

class Preprocess():
    """Custom feature extraction

    Feature extraction that generates features based on sentiment values and named entity recogntion among other things.
    """


    def __init__(self, *args, **kwargs):
        #Load in all required thing from packages.


        nltk.download('punkt')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')


        super(Preprocess, self).__init__(*args, **kwargs)

    def convert_lower_case(self, text):
        return np.char.lower(text)

    def remove_stop_words(self, text):
        stop_words = stopwords.words('english')
        words = word_tokenize(str(text))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
        return new_text

    def remove_punctuation(self, text):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in range(len(symbols)):
            text = np.char.replace(text, symbols[i], ' ')
            text = np.char.replace(text, "  ", " ")
        text = np.char.replace(text, ',', '')
        return text

    def remove_apostrophe(self,text):
        return np.char.replace(text, "'", "")

    def stemming(self,text):
        stemmer = PorterStemmer()

        tokens = word_tokenize(str(text))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
        return new_text

    def convert_numbers(self,text):
        tokens = word_tokenize(str(text))
        new_text = ""
        for w in tokens:
            try:
                w = num2words(int(w))
            except:
                a = 0
            new_text = new_text + " " + w
        new_text = np.char.replace(new_text, "-", " ")
        return new_text

    def preprocess(self, text):
        text = self.convert_lower_case(text)
        text = self.remove_punctuation(text)  # remove comma seperately
        text = self.remove_apostrophe(text)
        text = self.remove_stop_words(text)
        text = self.convert_numbers(text)
        text = self.stemming(text)
        text = self.remove_punctuation(text)
        text = self.convert_numbers(text)
        text = self.stemming(text)  # needed again as we need to stem the words
        text = self.remove_punctuation(text)  # needed again as num2word is giving few hypens and commas fourty-one
        text = self.remove_stop_words(text)  # needed again as num2word is giving stop words 101 - one hundred and one
        return text

    def doc_freq(self,word):
        c = 0
        try:
            c = DF[word]
        except:
            pass
        return c


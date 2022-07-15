import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from num2words import num2words
import nltk


nltk.download('punkt')
nltk.download('stopwords')

def convert_lower_case(text):
    return np.char.lower(text)

def remove_stop_words(text):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(text))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
             new_text = new_text + " " + w
    return new_text

def remove_punctuation(text):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        text = np.char.replace(text, symbols[i], ' ')
        text = np.char.replace(text, "  ", " ")
    text = np.char.replace(text, ',', '')
    return text

def remove_apostrophe(text):
    return np.char.replace(text, "'", "")

def stemming(text):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(text))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(text):
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

def preprocess(text):
    text = convert_lower_case(text)
    text = remove_punctuation(text)  # remove comma seperately
    text = remove_apostrophe(text)
    text = remove_stop_words(text)
    text = convert_numbers(text)
    text = stemming(text)
    text = remove_punctuation(text)
    text = convert_numbers(text)
    text = stemming(text)  # needed again as we need to stem the words
    text = remove_punctuation(text)  # needed again as num2word is giving few hypens and commas fourty-one
    text = remove_stop_words(text)  # needed again as num2word is giving stop words 101 - one hundred and one
    return text




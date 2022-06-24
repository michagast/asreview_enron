
from asreview.models.feature_extraction.base import BaseFeatureExtraction
import numpy as np

import pandas as pd                 #For data science purposes
import re                           #For performing regex
import torch                        #For running models with cude
import nltk.data                    #For various things
#import enchant                      #For BagOfWords feature
from sklearn.feature_extraction.text import CountVectorizer #For BagOfWords feature

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline

class Enron(BaseFeatureExtraction):
    """Custom feature extraction

    Feature extraction that generates features based on sentiment values and named entity recogntion among other things.
    """

    name = "enron"
    label = "Enron feature extraction"

    def __init__(self, *args, **kwargs):
        self._model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self._model.eval()
        self._tokenizernlp = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self._modelner = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english", return_dict=True)
        self._modelner.eval()
        self._tokenizerner = AutoTokenizer.from_pretrained('xlm-roberta-large-finetuned-conll03-english')
        self.alphabets = "([A-Za-z])"
        self.prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        self.suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        self.starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        self.acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        self.websites = "[.](com|net|org|io|gov)"
        #self.dictionary = enchant.Dict("en_US")
        self.vectorizer = CountVectorizer()


        super(Enron, self).__init__(*args, **kwargs)
    #Todo refactor this so that no for loop is used
    def transform(self, texts):
        resultsentiment = np.empty([0])
        resulttextlen = np.empty([0])
        resultspecificwords = np.empty([0])
        resultner = np.array([])
        resultstddevsentence = np.empty([0])
        #result_bow = np.array([])
        for text in texts:
            resultsentiment = np.append(resultsentiment, self.generatesentimentvalues(text))
            resulttextlen = np.append(resulttextlen, self.gettextlength(text))
            resultspecificwords = np.append(resultspecificwords, self.specific_words_check(text))
            resultner = np.append(resultner, self.generate_named_entities(text), axis = 0)
            resultstddevsentence = np.append(self.standard_dev_sentence_length(text), axis=0)
            #result_bow = np.append(result_bow, self.bag_of_words(text))
        resultner = resultner.reshape(int(len(resultner)/4),4)
        resultsentiment = resultsentiment.reshape(-1, 1)
        resulttextlen = resulttextlen.reshape(-1,1)
        resultspecificwords = resultspecificwords.reshape(-1,1)
        resultstddevsentence = resultstddevsentence.reshape((-1,1))
        result = np.hstack((resultsentiment, resulttextlen, resultspecificwords, resultstddevsentence,  resultner))

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
            r'(\b[Mm]+eet?(ing)?s?\b|\b[Pp]+lane\b|\bexpense report\b|\b[Cc]+all\b|\b[Vv]+oicemail\b|\b[Ee]+?[Mm]+ail(ing)?\b|\b[Ww]+eeks\b|\b([Ss]+chedul(e)?(ing)?)|\b[Tt]+ime|\b[Ww]+eek\b|\b[Ii]+nvite?d?(ing)?\b|\b([0-1]?[0-9]):[0-5][0-9]\b)', text))
        if amount_of_words:
            return (amount_of_words)
        else:
            return 0

    def split_into_sentences(self, text):
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(self.prefixes, "\\1<prd>", text)
        text = re.sub(self.websites, "<prd>\\1", text)
        if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub("\s" + self.alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(self.acronyms + " " + self.starters, "\\1<stop> \\2", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]" + self.alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + self.suffixes + "[.] " + self.starters, " \\1<stop> \\2", text)
        text = re.sub(" " + self.suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + self.alphabets + "[.]", " \\1<prd>", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

    def remove_short_tokens(self, tokens):
        average = 0
        for token in tokens:
            average += len(token)
        try:
            average = average / len(tokens)
            return ([x for x in tokens if len(x) >= average])
        except:
            return (tokens)
    #TO0DO:Remove All B-entities if they do not add anything useful to the model
    def generate_entity_list(self, entities):
        B_LOC, B_MISC, B_ORG, I_LOC, I_MISC, I_ORG, I_PER = 0, 0, 0, 0, 0, 0, 0
        for entity in entities:
            if entity == 'B-LOC':
                B_LOC += 1
            elif entity == 'B-MISC':
                B_MISC += 1
            elif entity == 'B-ORG':
                B_ORG += 1
            elif entity == 'I-LOC':
                I_LOC += 1
            elif entity == 'I-MISC':
                I_MISC += 1
            elif entity == 'I-ORG':
                I_ORG += 1
            elif entity == 'I-PER':
                I_PER += 1
        return ([I_LOC, I_MISC, I_ORG, I_PER])

    def generate_named_entities(self, text):
        ''' Function that generates named entity values for the inputted text.
        This Method does a few things. First it splits the text into single sentences(split_into_senteces). The short sentences are then removed based on the average length of the sentences in the text(remove_short_tokens)
        The tokens are then fed into a tokenizer and the generated tokens are fed into a model that generates named entities based on the tokens. The result of this is returned as a dict which can then be appended to the dataframe.
        them to the dataframe and removing the old one.
        '''
        tokens = [x for x in self.split_into_sentences(text) if not any(y in x for y in ['/','+'])]  # split text into sentences and remove any sentence that contains / or + as a character
        tokens = self.remove_short_tokens(tokens)
        if tokens:
            inputs = self._tokenizerner.batch_encode_plus(tokens, return_tensors="pt", padding=True, max_length=512, truncation=True)  # tokenize sentences, max_length is 512 for if cuda is enabled to speed the model up
            with torch.no_grad():
                results = self._modelner(**inputs)
                for i, _input in enumerate(inputs['input_ids']):
                    namedentities = [self._modelner.config.id2label[item.item()] for item in results.logits[i].argmax(axis=1)]  # for every probability for a named entity for a word, turn the probabilities into their associated labels
            entitynumberslist = self.generate_entity_list(namedentities)  # Based on the array of entity names that is generated, count each entity and make a dict of this
        else:
            entitynumberslist = [0,0,0,0]
        return entitynumberslist

    def remove_numbers_phonenumbers(self,text):
        text = re.sub(r'\b([0-9]{3}-[0-9]{3}-[0-9]{4})\b', '', text)
        text = re.sub(r'\b([0-1][0-9]\/[0-3][0-9]\/[0-9]{4})\b', '', text)
        text = re.sub(r'\b([0-1]?[0-9]):[0-5][0-9]\b', '', text)
        text = re.sub(r'\b\w*\d\w*\b', '', text)
        text = re.sub(r'\b ?[0-9]+ \b', '', text)
        splits = text.split()
        for word in splits:
            if not self.dictionary.check(word):
                text = text.replace(word, '')
        return text

    def bag_of_words(self, text):
        text = self.remove_numbers_phonenumbers(text)
        X_bow = self.vectorizer.fit_transform(text)
        df_bow = pd.DataFrame(X_bow.toarray(),columns=self.vectorizer.get_feature_names_out())
        try:
            df_bow.drop(['label'], axis=1, inplace=True)
        except:
            pass
        df_bow= df_bow[df_bow.sum(axis=0).sort_values(ascending=False)[0:100].index.values]
        return df_bow.to_numpy()

    def standard_dev_sentence_length(self,text):
        sentences = nltk.tokenize.sent_tokenize(text)
        sentence_length = []
        for item in sentences:
            sentence_length.append(len(item))
        return (np.std(sentence_length))



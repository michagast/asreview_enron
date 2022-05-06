
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
        self._model.eval()
        self._tokenizernlp = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self._modelner = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        self._modelner.eval()
        self._tokenizerner = AutoTokenizer.from_pretrained('xlm-roberta-large-finetuned-conll03-english')
        self.alphabets = "([A-Za-z])"
        self.prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        self.suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        self.starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        self.acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        self.websites = "[.](com|net|org|io|gov)"
        super(Enron, self).__init__(*args, **kwargs)
    #Todo refactor this so that no for loop is used
    def transform(self, texts):
        resultsentiment = np.empty([0])
        resulttextlen = np.empty([0])
        resultspecificwords = np.empty([0])
        resultner = np.array([])
        for text in texts:
            resultsentiment = np.append(resultsentiment, self.generatesentimentvalues(text))
            resulttextlen = np.append(resulttextlen, self.gettextlength(text))
            resultspecificwords = np.append(resultspecificwords, self.specific_words_check(text))
            resultner = np.append(resultner, self.generate_named_entities(text), axis = 0)
        resultner = resultner.reshape(int(len(resultner)/7),7)
        resultsentiment = resultsentiment.reshape(-1, 1)
        resulttextlen = resulttextlen.reshape(-1,1)
        resultspecificwords = resultspecificwords.reshape(-1,1)
        result = np.hstack((resultsentiment, resulttextlen, resultspecificwords, resultner))

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

    def generate_named_entities(self, text):
        ''' Function that generates named entity values for the inputted text.
        This Method does a few things. First it splits the text into single sentences(split_into_senteces). The short sentences are then removed based on the average length of the sentences in the text(remove_short_tokens)
        The tokens are then fed into a tokenizer and the generated tokens are fed into a model that generates named entities based on the tokens. The result of this is returned as a dict which can then be appended to the dataframe.
        them to the dataframe and removing the old one.
        '''
        tokens = [x for x in self.split_into_sentences(text) if not any(y in x for y in ['/','+'])]  # split text into sentences and remove any sentence that contains / or + as a character
        tokens = self.remove_short_tokens(tokens)
        if tokens:
            inputs = self._tokenizerner.batch_encode_plus(tokens, return_tensors="pt", padding=True, max_length=512,truncation=True)  # tokenize sentences, max_length is 512 for if cuda is enabled to speed the model up
            with torch.no_grad():
                results = self._modelner(**inputs)
                if results:
                    for i, input in enumerate(inputs['input_ids']):
                        namedentities = [self._modelner.config.id2label[item.item()] for item in results.logits[i].argmax(axis = 1)]  # for every probability for a named entity for a word, turn the probabilities into their associated labels
            entitynumberlist = self.generate_entity_list(namedentities)  # Based on the array of entity names that is generated, count each entity and make a dict of this
        else:
            entitynumberlist = [0,0,0,0,0,0,0]
        return entitynumberlist

    def remove_short_tokens(self, tokens):
        average = 0
        for token in tokens:
            average += len(token)
        try:
            average = average / len(tokens)
            return ([x for x in tokens if len(x) >= average])
        except:
            return (tokens)

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
        return ([B_LOC, B_MISC, B_ORG, I_LOC, I_MISC, I_ORG, I_PER])




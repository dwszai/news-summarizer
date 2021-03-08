# importing package
import os
import re
import pandas as pd
import sklearn
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer


def clean_text(text, remove_stopwords = True):
        """
        remove artifacts, unneccessary words etc
        """
        ## regex method - remove '\n'
        cleantext = re.sub(r"\\n", " ", text)
        ## remove '\BA'
        cleantext = re.sub(r"\\BA", " ", cleantext)
        ## remove '\'
        cleantext = re.sub(r"\\", " ", cleantext)
        ## remove parenthesis
        cleantext = re.sub(r'\([^)]*\)', '', cleantext)
        ## removing double spaces with single space
        cleantext = re.sub('\s\s+', " ", cleantext)
        # remove 'b''
        cleantext = re.sub(r"b'","",cleantext)
        ## substitute % with 'percent'
        cleantext = re.sub(r"[%]", "percent",cleantext)
        ## substitute $ with 'USD'
        cleantext = re.sub(r"[$]", "USD",cleantext)
        ## remove '''
        cleantext = re.sub(r"[\']", "", cleantext)
        ## remove '-'
        cleantext = re.sub(r"[-']", "", cleantext)
        ## remove '""'
        cleantext = re.sub(r"[\"]", "", cleantext)
        ## remove all b"
        cleantext = cleantext.strip('b"')
        ## remove xc2
        cleantext = re.sub(r"xc2", "", cleantext)
        # remove xa359m
        cleantext = re.sub(r"xa359m", "", cleantext)
        # remove xa35.7n 
        cleantext = re.sub(r"xa35.7n ", "", cleantext)
        # remove xa3160m
        cleantext = re.sub(r"xa3160m", "", cleantext)
        # remove xa35.7bn
        cleantext = re.sub(r"xa35.7bn ", "", cleantext)
        # remove xa3125m
        cleantext = re.sub(r"xa3125m", "", cleantext)
        # remove xa375m
        cleantext = re.sub(r"xa375m", "", cleantext)
        # remove xa3106m
        cleantext = re.sub(r"xa3106m", "", cleantext)
        # remove xa31.97bn
        cleantext = re.sub(r"xa31.97bn", "", cleantext)
        # remove xa3250m
        cleantext = re.sub(r"xa3250m", "", cleantext)
        # remove xa36
        cleantext = re.sub(r"xa36", "", cleantext)
        # remove xa310
        cleantext = re.sub(r"xa310", "", cleantext)
        # remove xa34
        cleantext = re.sub(r"xa34", "", cleantext)
        # remove xa32.50
        cleantext = re.sub(r"xa32.50", "", cleantext)
        # lower case
        cleantext = cleantext.lower()
    
    
        if remove_stopwords:
            cleantext = cleantext.split()
            stops = set(stopwords.words("english"))
            cleantext = [w for w in cleantext if not w in stops]
            cleantext = " ".join(cleantext)

        return cleantext

class DataPipeline:
    def __init__(self, datapath):
        self.data = pd.read_csv(datapath)

    def clean_news(self):
        """
        clean news column and add a 'cleannews' column
        """
        self.data['cleannews'] = self.data['news'].apply(lambda x: clean_text(x))
        self.data['cleansummary'] = self.data['summary'].apply(lambda x: clean_text(x))

        return self.data

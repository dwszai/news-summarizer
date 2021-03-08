import os
import re
import heapq
import requests
import pandas as pd
from newspaper import fulltext
import nltk
from bert_score import score
from rouge import Rouge

DATA_ORIGINAL = 'news'
DATA_SUMMARIZATION = 'summary'
SAMPLE_SIZE = 1000
RANDOM_SEED = 42

class ExtractiveSummarizer():
    def __init__(self):
        pass

    def load_url(self, url):
        """Load url link to generate article text

        Args:
            url (str): link of article

        Returns:
            text: string of text article from the url provided
        """
        text = fulltext(requests.get(url).text)

        return text

    def load_data(self, data_path):
        """Load data from a path

        Args:
            data_path (str): directory/path of dataset

        Returns:
            data: dataframe object
        """
        data = pd.read_csv(data_path)

        return data

    def _preprocessing(self, text):
        """simple preprocessing if needed

        Args:
            text (str): uncleaned text

        Returns:
            text: cleaned text
        """
        # Removing Square Brackets and Extra Spaces
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        # Removing special characters and digits
        formatted_text = re.sub('[^a-zA-Z]', ' ', text)
        formatted_text = re.sub(r'\s+', ' ', formatted_text)

        return text, formatted_text

    def _word_frequency(self, formatted_text):
        """Find the frequency of each word for weighing later

        Args:
            formatted_text (str): cleaned text

        Returns:
            word_frequencies: the frequencies of each word
        """
        stopwords = nltk.corpus.stopwords.words('english')
        word_frequencies = {}
        for word in nltk.word_tokenize(formatted_text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequncy = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy) 
        
        return word_frequencies

    def _sentence_score(self, text, word_frequencies):
        """Find the sentence score for each input text/doc

        Args:
            text (str): given text for weighing
            word_frequencies (dic): dictionary of words and their frequency (int)

        Returns:
            sentence_scores: the scoring given to each sentences
        """
        sentence_list = nltk.sent_tokenize(text)
        sentence_scores = {}
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]
        
        return sentence_scores
    
    def _weigh_sentence(self, data):
        """Compiling fullprocess for prediction later

        Args:
            data (str): input data

        Returns:
            sentence_scores: scoring given to each sentences
        """
        text, formatted_text = self._preprocessing(data)
        word_frequencies = self._word_frequency(formatted_text)
        sentence_scores = self._sentence_score(text, word_frequencies)
        
        return sentence_scores
        
    def predict(self, data):
        """Predict the summarized output given the data

        Args:
            data (str): input data

        Returns:
            summary: summarized version of the input data
        """
        sentence_scores = self._weigh_sentence(data)
        summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)

        return summary

    def evaluation(self, preds, refs):
        """Evaluate the performance of the model

        Args:
            preds (str): generated summary
            refs (str): input data/text

        Returns:
            scores: precision, recall, f1 scores to determine to performance of the model
        """
        precision, recall, f1_score = score(list(preds), list(refs), lang='en', verbose=True)
        rouge = Rouge()
        rouge_scores = rouge.get_scores(preds, refs, avg=True)

        return precision.mean(), recall.mean(), f1_score.mean(), rouge_scores

    def run(self, data_dir=r"data\bbc_news.csv"):
        """Execute all processes to generate scoring here

        Args:
            data_dir (str): data path. Defaults to r"data\bbc_news.csv".
        """
        data_path = os.path.join(os.getcwd(), data_dir)
        data = summarizer.load_data(data_path)

        reference_data = data[DATA_ORIGINAL].sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
        reference_summary = data[DATA_SUMMARIZATION].sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
        candidate_data = reference_data.map(summarizer.predict)

        # input_text = ""
        # summary = summarizer.predict(input_text)

        precision, recall, f1_score, rouge_scores = summarizer.evaluation(preds=candidate_data, refs=reference_summary)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1_score}')
        print(f'ROUGE Score: {rouge_scores}')

if __name__ == '__main__':
    summarizer = ExtractiveSummarizer()
    summarizer.run()
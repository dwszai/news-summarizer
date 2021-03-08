import os
import requests
import pandas as pd
from newspaper import fulltext
from summarizer import Summarizer
from bert_score import score
from rouge import Rouge

DATA_ORIGINAL = 'news'
DATA_SUMMARIZATION = 'summary'
SAMPLE_SIZE = 1000
RANDOM_SEED = 42

class BertModel():
    def __init__(self):
        self.model = None

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

    def load_model(self):
        """Load trained model

        Returns:
            model: pretrained model
        """
        self.model = Summarizer()

        return self.model

    def predict(self, input_text):
        """Generate a summary based on input text

        Args:
            input_text (str): input data/text for prediction

        Returns:
            summary: shorten version of the input text
        """
        result = self.model(input_text, min_length=30, max_length=500)
        summary = "".join(result)

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
        data = bert_model.load_data(data_path)

        reference_data = data[DATA_ORIGINAL].sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
        reference_summary = data[DATA_SUMMARIZATION].sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

        bert_model.load_model()
        candidate_data = reference_data.map(bert_model.predict)

        # input_text = ""
        # summary = bert_model.predict(input_text)

        precision, recall, f1_score, rouge_scores = bert_model.evaluation(preds=candidate_data, refs=reference_summary)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1_score}')
        print(f'ROUGE Score: {rouge_scores}')

if __name__ == '__main__':
    bert_model = BertModel()
    bert_model.run()

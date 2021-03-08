### import packages ###
import pandas as pd

from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import nltk
nltk.download('punkt')
from bert_score import score
import joblib


#---------------------------------------------------------------------------------------------------#


class ModelSumy():
    def __init__(self):
        self.model = None
        self.parser = None

    def load_data(self, file_path):
        """
        load data from path
        
        Args:
            file_path: str of file path
        Returns:
            data: dataframe of data
        """
        data = pd.read_csv(file_path)
        return data

    def train(self):
        """
        train model
        
        Args:
            input_text: str of input text
        Returns:
            model: initialised model
            parser: parser
        """
        self.model = KLSummarizer()
        
        # # save model
        # joblib.dump(self.model, r"F:\AIAP\all-assignments\team4\src\modelling\saved_models\model_sumy.sav")


    def predict(self, input_text):
        """
        predict summary
        
        Args:
            model: initialised model
            parser: parser
        Returns:
            pred: predicted summary
        """
        self.parser = PlaintextParser.from_string(input_text,Tokenizer('english'))
        kl_summary = self.model(self.parser.document, sentences_count=5)
        pred = '. '.join([str(sent) for sent in kl_summary])

        return pred

    def eval(self, preds, refs):
        """
        predict summary
        
        Args:
            model: initialised model
            tokenizer: tokenizer
            input_text: text to use for prediction
        Returns:
            pred: predicted summary
        """
        precision, recall, f1 = score(list(preds), list(refs), lang='en', verbose=True)

        return precision.mean(), recall.mean(), f1.mean()


if __name__ == '__main__':
    model_sumy = ModelSumy()

    file_path = r"F:\AIAP\all-assignments\team4\data\data.csv"
    data = model_sumy.load_data(file_path)

    input_text = data['news'][0]

    model_sumy.train()

    pred = model_sumy.predict(input_text)

    print(pred)

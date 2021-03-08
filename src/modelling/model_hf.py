### import packages ###
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import AutoModelWithLMHead, AutoTokenizer
import pandas as pd
from bert_score import score


#---------------------------------------------------------------------------------------------------#


class ModelHF():
    def __init__(self):
        self.model = None
        self.tokenizer = None

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
            None
        Returns:
            None
        """
        self.model = T5ForConditionalGeneration.from_pretrained('mrm8488/t5-base-finetuned-summarize-news')
        self.tokenizer = T5Tokenizer.from_pretrained('mrm8488/t5-base-finetuned-summarize-news')

    def preprocess(self, text, tokenizer):
        """
        preprocess text for model
        
        Args:
            text: str to preprocess
            tokenizer: tokenizer
        Returns:
            preprocessed_text: preprocessed str
        """
        preprocessed_text = text.strip().replace("\n","")
        preprocessed_text = "summarize: " + preprocessed_text
        
        preprocessed_text = tokenizer.encode(preprocessed_text, return_tensors="pt")            

        return preprocessed_text                           

    def predict(self, input_text):
        """
        predict summary
        
        Args:
            model: initialised model
            tokenizer: tokenizer
            input_text: text to use for prediction
        Returns:
            pred: predicted summary
        """

        preprocessed_text = input_text.strip().replace("\n","")
        preprocessed_text = "summarize: " + preprocessed_text
        
        preprocessed_text = self.tokenizer.encode(preprocessed_text, return_tensors="pt")            
        
        summary_ids = self.model.generate(preprocessed_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=2,
                                            min_length=30,
                                            max_length=100,
                                            early_stopping=True)
        
        pred = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)             

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
    model_hf = ModelHF()

    file_path = r"F:\AIAP\all-assignments\team4\data\data.csv"
    data = model_hf.load_data(file_path).sample(5, random_state=42).reset_index(drop=True)

    model_hf.train()

    # input_text = data['news'].apply(lambda x: preprocess(x, tokenizer))
    pred = model_hf.predict(data['news'][0])
    
    # for observation only
    data['pred'] = data['news'].apply(lambda x: model_hf.predict(x))

    print(pred)

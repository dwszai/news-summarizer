### import packages ###
import argparse
import logging

import numpy as np
import joblib
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import nltk
nltk.download('punkt')
import re


#---------------------------------------------------------------------------------------------------#


def preprocess(text):
    """
    preprocess text

    Args:
        text: input text to preprocess
    Returns:
        output_text: preprocessed text
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
    cleantext = re.sub(r"(b')","",cleantext)
    ## substitute % with ' percent'
    cleantext = re.sub(r"[%]", " percent",cleantext)
    ## substitute $ with 'USD'
    cleantext = re.sub(r"[$]", "USD",cleantext)
    ## remove '\'
    cleantext = re.sub(r"[\\]", "", cleantext)
    ## remove '-'
    cleantext = re.sub(r"(-')", " ", cleantext)
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
    
    return cleantext

def load_model(model_path):
    """
    load trained model from path 
    
    Args:
        model_path: path to saved model
    Returns:
        model: loaded model
    """
    logging.info(model_path)
    model = joblib.load(model_path)

    return model

def get_summary(model, input_text):
    """
    predict summary with sumy
    
    Args:
        model: trained model
        input_text: text to summarise
    Returns:
        output_summary: summarised text
    """
    parser = PlaintextParser.from_string(input_text,Tokenizer('english'))
    kl_summary = model(parser.document, sentences_count=5)
    output_summary = ' '.join([str(sent) for sent in kl_summary])

    return output_summary

# def get_summary_t5(model, input_text):
#     """
#     predict summary with t5
    
#     Args:
#         model: trained model
#         input_text: text to summarise
#     Returns:
#         output_summary: summarised text
#     """
#     tokenizer = T5Tokenizer.from_pretrained('mrm8488/t5-base-finetuned-summarize-news')

#     preprocessed_text = input_text.strip().replace("\n","")
#     preprocessed_text = "summarize: " + preprocessed_text
    
#     preprocessed_text = tokenizer.encode(preprocessed_text, return_tensors="pt")            

#     summary_ids = model.generate(preprocessed_text,
#                                         num_beams=4,
#                                         no_repeat_ngram_size=2,
#                                         min_length=30,
#                                         max_length=100,
#                                         early_stopping=True)
    
#     output_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)             

#     return output_summary


if __name__ == '__main__':
    # read in args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--text')

    args = parser.parse_args()

    model_path = args.model_path
    text = args.text

    # preprocess text
    text = preprocess(text)

    # load model
    model = load_model(model_path)
    print('model loaded')

    # preds
    summary = get_summary(model, text)

    print('Predicted summary')
    print('===========================')
    print(summary)

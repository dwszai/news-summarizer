# Introduction

NEWSY


    team4
    ├── conda.yml
    ├── Dockerfile
    ├── init.sh
    ├── LICENCE
    ├── Procfile
    ├── README.md
    ├── requirements.txt
    ├── skaffold.yaml
    ├── ci
    ├── notebooks
    │   ├── A8_EDA_cleaning.ipynb
    │   ├── BertModel.ipynb
    │   ├── eda.ipynb
    │   └── ExtractiveModel.ipynb
    ├── scripts
    │   └── start_app.sh
    ├── polyaxon
    │   ├── experiment.yml
    │   ├── notebook.yml
    │   └── docker
    ├── img
    ├── src
    │   ├── app.py
    │   ├── helper.py
    │   ├── sample_summariser.py
    │   ├── templates
    │   ├── static
    │   ├── datapipeline
    │   │   ├── loader.py
    │   │   └── preprocess.py
    │   ├── experiment
    │   │   └── main.py
    │   └── modelling
    │   │   ├── saved_models
    │   │   └── unused_models
    └── tests
        └── __init__.py


Write an introduction to your project in this 
section and describe what it does.

- News articles of BBC from 2004 to 2005
- Created using dataset used for data categorization 
- Contains 2225 documents of news articles and their summaries
- In .txt format, which was converted to .csv format
- Involves 5 categories of news: Sport (511 documents), Business (510 documents), Politics (417 documents), Technology (401 documents), Entertainment (336 documents)

### 1b. Key EDA findings

#### **Summary Length Distribution & News Length Distribution**
![Length_distribution](src/static/img/summary_news_length_distribution1.png "Length distribution")

The length of the summary given in the dataset are mostly below 500 words, while the news article are mostly below 1000 words.

#### **N-gram**
N-gram model predicts occurence of word based on the occurence of its N-1 previous words. Some of the words that were grouped together were:

![ngram](src/static/img/n-gram11.png "ngram1") ![ngram](src/static/img/n-gram22.png "ngram2")


Some of the words that are found together that makes sense are (last,year), (mobile,phone), (new,york), etc.

#### **Word2vec**
Some exploration has also been done using word embedding with Word2Vec model.

![word2vec](src/static/img/word22vec.png "word2vec")

#### **Data Cleaning** 
- remove '\n', '\BA', parenthesis, multiple spaces, artifacts (e.g. '\'), various 'xa-' and 'xc2' that has not much meaningful insights
 - substituted '$' with USD as dollar signs here in BBC news refers to US Dollars
 - substituted '%' with 'percent'
 - remove stopwords


## 2. Model

4 models were tested for this project. Details of each are below.

1. NLTK Summarizer:
- A simple NLTK extractive summarizer that is used as our baseline model for comparison against the better pretrained models
- Uses weightage scores of each words to determine the scores of each sentence in the summary to generate a ranking for each sentence before output
- Summary will be generated based on the scoring of each sentences and the number of sentences chosen can be adjusted

2. Bert Model: 
- A pretrained BERT summarizer model that was trained on SciBERT(BERT model trained on scientific text)
- The training corpus was papers taken from Semantic Scholar. Corpus size is 1.14M papers, 3.1B tokens. Full text of the papers in training (not just abstracts)
- Additional layers can be added and trained in the future to further improve the evaluation score

3. KL-Sum Model (extractive):
- Selects sentences based on similarity of a sentence's word distribution against the full text
- Greedy algorithm
- Seeks to minimise the KL (Kullback–Leibler) divergence

4. T5 Transformer Model (abstractive):
- Pretrained on C4 (Colossal Clean Crawled Corpus), 700GB
- T5 base model with LM on top was used, finetuned on a news summary dataset

### 2a. Expected Format the model Requires
All of the 4 models requires text data as input and generate summarized text data as output. Preprocessing of the data will be done before being passed through the model.

### 2b. Details about the model Performance
We use bertscore evaluation to determine to scoring metrics for our model performance. The 3 scoring metrics are precision, recall and f1 score. The generated summary is evaluated against the original text provided, using contextual embeddings for the tokens. Cosine similarity is then used to compute the matching, optionally weighted with idf.

Below are the results for our 4 models:

__Bertscore__

                      precision    recall  f1-score

     NLTK Summarizer       0.93      0.86      0.89
                Bert       0.96      0.86      0.91
              KL-Sum       0.93      0.87      0.90
      T5 Transformer       0.93      0.83      0.88

__ROUGE-2__

                      precision    recall  f1-score

     NLTK Summarizer       0.46      0.39      0.41
                Bert       0.45      0.28      0.34
              KL-Sum       0.44      0.30      0.34
      T5 Transformer       0.38      0.19      0.24

__ROUGE-L__

                      precision    recall  f1-score

     NLTK Summarizer       0.55      0.51      0.52
                Bert       0.57      0.41      0.47
              KL-Sum       0.57      0.41      0.46
      T5 Transformer       0.57      0.34      0.42

__Findings__
- Results for all models are all comparable
- Output from NLTK model is quite similar to the reference summary, which could be a reason why the NLTK model has higher ROUGE scores
- T5 transformer has lower ROUGE scores, possibly due to a max token length of 512 i.e. for texts with >512 tokens it will be truncated
- KL-Sum model was chosen because of its speed in delivering results


## 3. How the model is served
Among the 4 models, we chose the KL-Sum model to use for our web. The model will be served on our Flask app to generate a summary of any given input text from the user. We will be using Heroku to host our app.

### 3a. Deployment steps
This uses a free template from [Start Bootstrap](https://startbootstrap.com/themes), Flask Backend, and deploy to Heroku PaaS.

There will be an app.py file created for the app deployment. Within it, the chosen model is called for generating the summary.
The predict function is to receive input text, call the model to summarize the input text, generates the summary and their respective word length. 

Example code:


```
@app.route('/', methods=['GET', 'POST'])

def predict():
    """Return Cell with Summary"""
    global CLEAN_SUMMARY
    summary = CLEAN_SUMMARY
    in_count = None
    out_count = None
    app.logger.info('Received Text Input')
    if request.method == 'POST':
        out = request.form['rawtext']
        summary = nltk_summarizer(out)
        in_count = len(request.form['rawtext'])
        out_count = len(summary.split())
    input_count_words = f"{in_count} words."
    output_count_words = f"{out_count} words."
    CLEAN_SUMMARY = summary
    return render_template('index.html', input_count=input_count_words, output=summary, output_count=output_count_words)
```

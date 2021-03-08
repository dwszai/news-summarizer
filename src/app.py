import os
import logging
import requests
import pandas as pd
from pathlib import Path
from waitress import serve
import markdown.extensions.fenced_code
from pygments.formatters.html import HtmlFormatter
from flask import Flask, render_template, request, make_response

# Local Functions
from src.helper import relevant_news, news_dict
# from src.sample_summariser import nltk_summarizer
from src.inference import preprocess, load_model, get_summary

app = Flask(__name__)

logging.basicConfig(format='[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s', level=logging.INFO)
logger = logging.getLogger('app')

logger.info('Server started')

# DEFINITIONS
# Source from news api
URL_NEWSAPI = ('http://newsapi.org/v2/top-headlines?'
               'country=sg&'
               'excludeDomains=bloomberg.com,scitechdaily.com&'
               'pageSize=100&'
               'apiKey=71511452c03541de8aee95f319f20a2f')

# Source from Mediastack.com
# URL_MEDIASTACK = ('http://api.mediastack.com/v1/news'
#                    '?access_key=aba78c86798394c9c47915095dfeafa3'
#                    '&countries=sg,us'
#                    '&sources=-bloomberg'
#                    '&limit=100')

# Root directory
ROOT = Path(__file__).parent.parent

# DATA STORAGE
CLEAN_ARTICLES = []
CLEAN_ARTICLES_DICT = {}
CLEAN_SUMMARY = None

# load model
relative_path_1 = 'src/modelling/saved_models/model_sumy.sav'
src_path_1 = (ROOT / relative_path_1).resolve()
model = load_model(src_path_1)

@app.route('/')
def index():
    """Homepage with empty Info"""
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    """Return Cell with Summary"""
    global CLEAN_SUMMARY
    global model

    summary = CLEAN_SUMMARY
    in_count = None
    out_count = None

    logger.info('Received Text Input')
    if request.method == 'POST':
        out = request.form['rawtext']

        # preprocess text
        text = preprocess(out)
        logger.info('text preprocessed')

        # get summary
        summary = get_summary(model, text)
        logger.info('obtained summary')

        in_count = len(request.form['rawtext'].split())
        out_count = len(summary.split())

    input_count_words = f"{in_count} words."
    output_count_words = f"{out_count} words."

    CLEAN_SUMMARY = summary

    return render_template('index.html', input_count=input_count_words, output=summary, output_count=output_count_words)

@app.route('/topten')
def topten():
    """Top Ten News"""
    global CLEAN_ARTICLES
    global CLEAN_ARTICLES_DICT
    cleaned_articles = CLEAN_ARTICLES

    url = URL_NEWSAPI
    response = requests.get(url)

    # Get top 10 articles
    logger.info('Retrieved articles')
    articles = response.json()['articles']

    # For Mediastack
    # articles = response.json()['data'][:10]

    # If haven't extract the info yet
    if CLEAN_ARTICLES == []:
        cleaned_articles = relevant_news(CLEAN_ARTICLES, articles, model)
        CLEAN_ARTICLES = cleaned_articles
        CLEAN_ARTICLES_DICT = news_dict(cleaned_articles)

    return render_template('topten.html', parsed_article=cleaned_articles)

@app.route('/csv_summary')
def csv_summary():
    """Download summary as csv file"""
    global CLEAN_SUMMARY

    to_csv = CLEAN_SUMMARY
    logger.info(to_csv)
    # If there is no summary
    if to_csv == None:
        return ('', 204)
    response = make_response(to_csv)
    cd = 'attachment; filename=summary.csv'
    response.headers['Content-Disposition'] = cd
    response.mimetype ='text/csv'

    return response

@app.route('/csv_news')
def csv_news():
    """Download Top 10 News as CSV file"""
    global CLEAN_ARTICLES_DICT

    to_csv = CLEAN_ARTICLES_DICT
    logger.info(to_csv.keys())
    # If there is no articles
    if to_csv == []:
        return ('', 204)
    else:
        keys = []
        vals = []
        for header, content in to_csv.items():
                keys.append(header)
                vals.append(content)
        result = pd.DataFrame({'Headline': keys, 'Summary': vals})

        response = make_response(result.to_csv())
        cd = 'attachment; filename=news.csv'
        response.headers['Content-Disposition'] = cd
        response.mimetype = 'text/csv'

        return response

@app.route('/docs')
def readme():

    file_path = (ROOT / 'README.md').resolve()

    # file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README.md')

    logger.info(file_path)
    # CSS for read me
    formatter = HtmlFormatter(style="emacs", full=True, cssclass="codehilite")
    css_string = formatter.get_style_defs()

    # Extract readme
    readme_file = open(file_path, "r")
    md_template_string = markdown.markdown(
        readme_file.read(), extensions=["fenced_code"]
    )
    # Combine both
    md_css_string = "<style>" + css_string + "</style>"
    md_template = md_css_string + md_template_string
    return render_template('readme.html', text=md_template)


if __name__ == "__main__":
    app.run(debug=True, port=8000)

    # Use gunicorn for serving in production
    # For production mode, comment the line above and uncomment below
    # serve(app, host="0.0.0.0", port=8000)
    logger.debug('A value for debugging')

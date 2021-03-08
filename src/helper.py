import logging
import requests
from newspaper import fulltext

# Local Function
# from src.sample_summariser import nltk_summarizer
from src.inference import preprocess, get_summary

def relevant_news(cleaned_articles, articles, model):
    """
    Extract text content from url websites and input into list.
    :param cleaned_articles: Empty list to store relevant news.
    :param articles: List of data of news articles.
    :return: cleaned_articles that contains top 10 relevant news content.
    """
    news = 0
    target_news = 0
    while(target_news<10):
        logging.info(f"Retrieved articles {news+1}/{len(articles)}]")
        # Retrieving text via url
        try:
            # Get Text from news article
            temp = fulltext(requests.get(articles[news]['url']).text)

            # preprocess text
            text = preprocess(temp)

            # get summary
            foo = get_summary(model, text)

            if foo != None or foo != '':
                articles[news]['content'] = foo
                cleaned_articles.append(articles[news])
                target_news += 1
            articles[news]['content'] = foo
            news += 1
        except:
            logging.info('News site deny parsing of article.')
            news += 1
        if news == 1000:
            # Break out of while loop to stop stack overflow
            cleaned_articles.append(articles[''])
            break


    return cleaned_articles

def news_dict(cleaned_articles):
    """
    For the cleaned_articles data, extract only the headline and summary.
    :param cleaned_articles: List of dictionaries, extract only the target information.
    :return: dictionary of Headlines to Summary.
    """
    temp_dict = {}
    for i in cleaned_articles:
        temp_dict[i['title']] = i['content']
    return temp_dict
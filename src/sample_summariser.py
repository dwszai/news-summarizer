import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq

# Import NLTK corpus and tokenizer
# from static/nltk_data import punkt, stopwords

def check_empty(text):
	if not text:
		print('The text to be tokenized is a None type. Defaulting to blank string.')
		text = ''
	return text


def nltk_summarizer(raw_text):
	try:
		nltk.data.find('app/nltk_data/')
	except:
		nltk.download('stopwords')
		nltk.download('punkt')
	# nltk.data.find('~\\static\\nltk_data\\punkt.zip')
	# nltk.data.find('~\\static\\nltk_data\\stopwords.zip')
	# nltk.data.path.append('src/static/nltk_data/')
	stopWords = set(stopwords.words("english"))
	word_frequencies = {}
	# Remove empty strings to prevent errors
	for word in nltk.word_tokenize(check_empty(raw_text)):
	    if word not in stopWords:
	        if word not in word_frequencies.keys():
	            word_frequencies[word] = 1
	        else:
	            word_frequencies[word] += 1

	# Remove empty lists to prevent errors
	# wf_values = [x for x in word_frequencies.values() if x != []]
	maximum_frequency = max(word_frequencies.values(), default=0)

	for word in word_frequencies.keys():
	    word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

	sentence_list = nltk.sent_tokenize(check_empty(raw_text))
	sentence_scores = {}
	for sent in sentence_list:
	    for word in nltk.word_tokenize(sent.lower()):
	        if word in word_frequencies.keys():
	            if len(sent.split(' ')) < 30:
	                if sent not in sentence_scores.keys():
	                    sentence_scores[sent] = word_frequencies[word]
	                else:
	                    sentence_scores[sent] += word_frequencies[word]



	summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

	summary = ' '.join(summary_sentences)
	return summary
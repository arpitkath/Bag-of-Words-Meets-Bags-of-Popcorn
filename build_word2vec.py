# -*- coding: utf-8 -*- 
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
from gensim.models import word2vec
import string

train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3, encoding="utf-8" )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3, encoding="utf-8" )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3, encoding="utf-8" )

def clean_query(s):
	#s = re.sub(r'[^\w]', '', s)
	#s = re.sub(r'[^\x00-\x7F]+','', s)
	return ''.join([i if ord(i) < 128 else ' ' for i in s])

print "Data loaded."
# Will return a list of words in the sentence after removing HTML tags, non alphabets, and converting them to lower case.
def review2list( review ):

	text = BeautifulSoup(review, 'html').get_text().strip()
	text = re.sub("[^a-zA-Z]", "", text)
	words = text.lower().split()
	stop_words = set(stopwords.words("english"))
	words = [word for word in words if word not in stop_words]
	return words

# Will return a list of sentences extracted from paragraph.
def review2sentences( review, tokenizer ):

	_sentences = tokenizer.tokenize(review.strip())
	sentences = []
	for sentence in _sentences:
		if len(sentence) > 0:
			sentences.append(review2list(sentence))

	return sentences

sentences = []
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
print "Chunking sentences."
for sentence in train["review"]:
	sentences.extend(review2sentences(clean_query(sentence), tokenizer))

for sentence in unlabeled_train["review"]:
	sentences.extend(review2sentences(sentence, tokenizer))
print len(sentences), "sentences loaded."
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

model = word2vec.Word2Vec(sentences, workers=4, size=200, min_count=50, window=10, sample=1e-3)
model.init_sims(replace=True)
model.save("bowmbop_word2vec")

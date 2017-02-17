import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
from gensim.models import word2vec

train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

print "Data loaded."
# Will return a list of words in the sentence after removing HTML tags, non alphabets, and converting them to lower case.
def review2list( review ):

	text = BeautifulSoup(review).get_text().decode('utf-8').strip()
	text = re.sub("[^a-zA-Z]", "", text)
	words = text.lower().split()
	stop_words = set(stopwords.words("english"))
	words = [word for word in words if word not in stop_words]
	return words

# Will return a list of sentences extracted from paragraph.
def review2sentences( review, tokenizer ):

	_sentences = tokenizer.tokenize(review.decode('utf-8').strip())
	sentences = []
	for sentence in sentences:
		if len(sentence) > 0:
			sentences.append(review2list(sentence))

	return sentences

sentences = []
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
print "Chunking sentences."
for sentence in train["review"]:
	sentences.extend(review2sentences(sentence, tokenizer))

for sentence in unlabeled_train["review"]:
	sentences.extend(review2sentences(sentence, tokenizer))

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

model = word2vec.Word2Vec(sentences, workers=2, size=200, min_count=50, window=10, downsampling=1e-3)
model.init_sims(replace=True)
model.save("bowmbop_word2vec")

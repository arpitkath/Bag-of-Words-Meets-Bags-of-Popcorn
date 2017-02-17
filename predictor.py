from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load("bowmbop_word2vec")

def sentence2vec(words, model, num_features):

	word_vec = np.zeros((num_features, ), dtype="float32")
	n_words = 0
	vocab = set(model.index2word)

	for word in words:
		if word in vocab:
			word_vec = np.add(word_vec, model[word])
			n_words += 1

	word_vec = np.divide(word_vec, n_words)
	return vec

def make_avg_vec(reviews, model, num_features):

	feature_vec = np.zeros((len(reviews), num_features), dtype="float32")
	count = 0

	for review in reviews:

		if count % 1000 == 0:
			print "%d done." % (count)
		feature_vec[count] = sentence2vec(review, model, num_features)
		count += 1

	return feature_vec

# Will return a list of words in the sentence after removing HTML tags, non alphabets, and converting them to lower case.
def review2list( review ):

	text = BeautifulSoup(review).get_text()
	text = re.sub("[^a-zA-Z]", "", text)
	words = text.lower().split()
	stop_words = set(stopwords.words("english"))
	words = [word for word in words if word not in stop_words]
	return words

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review2list(review) )

trainDataVecs = make_avg_vec( clean_train_reviews, model, num_features )

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review2list(review) )

testDataVecs = make_avg_vec( clean_test_reviews, model, num_features )

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

forest = forest.fit( trainDataVecs, train["sentiment"] )

result = forest.predict( testDataVecs )

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
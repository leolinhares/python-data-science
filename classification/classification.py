import xml.etree.ElementTree as etree
# from lxml import etree
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import RSLPStemmer
from nltk import FreqDist
from nltk import NaiveBayesClassifier
from nltk import DecisionTreeClassifier
import nltk.classify
import math
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC

tree = etree.parse('news_data.xml')
root = tree.getroot()

# remove special caracters whilst tokenizing
tokenizer = RegexpTokenizer(r'\w+')

# portuguese stemmer
stemmer = RSLPStemmer()

all = []
features = []
labels = []

it = tree.iter(tag='item')
stops = set(stopwords.words('portuguese'))
for elem in it:
    all_text = ""

    # label
    category = elem.get('category')
    labels.append(category)

    # features
    channel = elem.get('channelName')
    title = elem.find("title").text
    description = elem.find("description").text
    text = elem.find("text").text

    all_text = ' '.join(filter(None, (channel, title, description, text)))

    all_text = tokenizer.tokenize(all_text)

    filtered_words = [word for word in all_text if word.lower() not in stops and len(word.lower()) > 1]

    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    features.append(stemmed_words)

    all += stemmed_words

fd = FreqDist(all)
most_common_words = [word_tuple[0] for word_tuple in fd.most_common(200)]
featuresets = []
print(most_common_words)
for item, label in zip(features, labels):
    selected_features = {}
    for word in most_common_words:
        selected_features[word] = word in item
    featuresets.append((selected_features, label))


size = len(features)
nt = int(math.floor(size * 0.7))
random.shuffle(featuresets)
train_set, test_set = featuresets[:nt], featuresets[nt:]

NB_classifier = NaiveBayesClassifier.train(train_set)
print("NB_classifier accuracy percent:", (nltk.classify.accuracy(NB_classifier, test_set))*100)
print(NB_classifier.show_most_informative_features(30))

DT_classifier = DecisionTreeClassifier.train(train_set)
print("DT_classifier accuracy percent:", (nltk.classify.accuracy(DT_classifier, test_set))*100)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, test_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, test_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(train_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, test_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(train_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, test_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)

# def main():
#     pass
# if __name__ == "__main__":
#     main()

import xml.etree.ElementTree as etree
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import RSLPStemmer
from nltk import FreqDist
from nltk import NaiveBayesClassifier
from nltk import DecisionTreeClassifier
from nltk import classify
import math
import random

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
print(size)
nt = int(math.floor(size * 0.7))
random.shuffle(featuresets)
train_set, test_set = featuresets[:nt], featuresets[nt:]
classifier_naive = NaiveBayesClassifier.train(train_set)
print(classify.accuracy(classifier_naive, test_set))
print(classifier_naive.show_most_informative_features())
classifier_decisiontree = DecisionTreeClassifier.train(train_set)
print(classify.accuracy(classifier_decisiontree, test_set))


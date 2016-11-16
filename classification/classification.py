import xml.etree.ElementTree as etree
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import RSLPStemmer
from nltk import FreqDist
tree = etree.parse('news_data.xml')
root = tree.getroot()

# remover caracteres especiais
tokenizer = RegexpTokenizer(r'\w+')
# portuguese stemmer
stemmer = RSLPStemmer()

features = []
it = tree.iter(tag='item')
for elem in it:
    all_text = ""

    #label
    category = elem.get('category')

    #features
    channel = elem.get('channelName')
    title = elem.find("title").text
    description = elem.find("description").text
    text = elem.find("text").text

    # Useless data
    # print elem.find("pubDate").text
    # print elem.find("when").text

    all_text = ' '.join(filter(None, (channel, title, description, text)))
    all_text = tokenizer.tokenize(all_text)
    filtered_words = [stemmer.stem(word) for word in all_text if word not in stopwords.words('portuguese')]

    fd = FreqDist(filtered_words)
    features.append(fd.most_common(10))

print(features[:10])



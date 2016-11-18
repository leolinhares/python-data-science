# Text Classification
### Python version
* Python3 (although it will work on Python2) 

### Required Python Libraries
* nltk
* numpy
* scipy
* sklearn

### Instructions
* Install nltk data
`python -m nltk.downloader all`
* Execute the file
`python classification.py`

### Discussion
The data was preprocessed using the Natural Language Toolkit (NLTK).  The following steps were taken during the process:
1. Tokenization using a RegexpTokenizer to remove punctuation
2. Portuguese stopwords removal
3. Stemming using RSLPStemmer (for Portuguese only)

After the preprocessing step, the features were extracted and selected based on the section Document Classification of the NLTK book. [ [6. Learning to Classify Text](http://www.nltk.org/book/ch06.html) ] The technique is called Bag of Words. A list of the 200 most frequent words in the dataset were constructed and the feature extractor simply checks whether the word is present in each given data item. 

Finally the data was randomized and split between train and test sets. It was used 8 classifiers (from nltk and sklearn) to compare the accuracy.

### Results
The script takes approximately 1 minute to execute.  It is printed the 200 most common words on the document and then the accuracy of each classifier (Naive Bayes, Decision Tree, Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, Stochastic Gradient Descent Classifier, Support Vector Classification , Linear Support Vector Classification). It is also printed the most relevant features of the Naive Bayes classifier.
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


#define the categories of the data
categories = [
     'alt.atheism',
     'talk.religion.misc',
     'comp.graphics',
     'sci.space',
     'rec.sport.hockey'
              ]

#calling the data
data_train = fetch_20newsgroups(subset='train',categories=categories, random_state=42)
data_test = fetch_20newsgroups(subset='test',categories=categories, random_state=42)

names = set(names.words())
lem = WordNetLemmatizer()
#prepare functions to clean the data please check spam mails with Naive Bayes to understand the cleaning opearation
def letter_only(astr):

    return astr.isalpha()


def clean_txt (docs):

    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lem.lemmatize(word.lower()) for word in doc.split()
                                     if letter_only(word) and word not in names]))
    return cleaned_docs

#clean the data
cleaned_train = clean_txt(data_train.data)
label_train = data_train.target
cleaned_test = clean_txt(data_test.data)
label_test = data_test.target
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=0.5, stop_words='english', max_features=8000)
term_docs_train = tfidf_vectorizer.fit_transform(cleaned_train)
term_docs_test = tfidf_vectorizer.transform(cleaned_test)


#preparing the algorithm
svm = SVC(kernel='linear', random_state= 42)

#fit the algorithm, make prediction and evaluate the model
svm.fit(term_docs_train, label_train)
accuracy = svm.score(term_docs_test, label_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))
prediction = svm.predict(term_docs_test)
report = classification_report(label_test , prediction)
print(report)
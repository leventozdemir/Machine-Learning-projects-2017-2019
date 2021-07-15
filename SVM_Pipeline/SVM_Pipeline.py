import timeit
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

lem = WordNetLemmatizer()
names = set(names.words())
def letter_only(astr):

    return astr.isalpha()

def clean_text (docs):

    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lem.lemmatize(word.lower()) for word in doc.split()
                                     if letter_only(word) and word not in names]))
    return cleaned_docs

categories = None
data_train = fetch_20newsgroups(subset='train',categories=categories, random_state=42)
data_test = fetch_20newsgroups(subset='test',categories=categories, random_state=42)
cleaned_train = clean_text(data_train.data)
label_train = data_train.target
cleaned_test = clean_text(data_test.data)
label_test = data_test.target
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=0.5, stop_words='english', max_features=8000)
term_docs_train = tfidf_vectorizer.fit_transform(cleaned_train)
term_docs_test = tfidf_vectorizer.transform(cleaned_test)
start_time = timeit.default_timer()
svc_pip = Pipeline([('tfidf' , TfidfVectorizer(stop_words='english')),
                    ('svc' , LinearSVC()),])
parameters_pipeline = {
     'tfidf__max_df': (0.25, 0.5),
     'tfidf__max_features': (40000, 50000),
     'tfidf__sublinear_tf': (True, False),
     'tfidf__smooth_idf': (True, False),
     'svc__C': (0.5, 1),
                        }
grid_search_pip = GridSearchCV(svc_pip, parameters_pipeline, n_jobs=-1, cv=3)
grid_search_pip.fit(cleaned_train,label_train)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))
print('params_pip = ', grid_search_pip.best_params_)
print('score_pip = ', grid_search_pip.best_score_)
svc_pip_best= grid_search_pip.best_estimator_
accuracy_pip= svc_pip_best.score(cleaned_test, label_test)
print('The accuracy_pip on testing set is: {0:.1f} %'.format(accuracy_pip*100))

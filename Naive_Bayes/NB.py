

# read all of the email text files and keep the ham/spam class information
#in the label variable where 1 represents spam email and 0 otherwise.
import os , glob

file_spm = 'enron1/spam'
file_ham = 'enron1/ham'

mails , labels = []  , []


for FN in glob.glob(os.path.join(file_ham , '*.txt')):
    with open(FN , 'r' , encoding="ISO-8859-1") as infile:
        mails.append(infile.read())
        labels.append(0)


for filename in glob.glob(os.path.join(file_spm, '*.txt')):
    with open(filename, 'r', encoding = "ISO-8859-1") as infile:
      mails.append(infile.read())
      labels.append(1)

print(len(mails))
print(len(labels))

#preprocess and clean the raw text data
#Number and punctuation removal, Human name removal, Stop words removal, Lemmatization
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

names = set(names.words())
lem = WordNetLemmatizer()

def letter_only(astr):

    return astr.isalpha()

def clean_txt (docs):

    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lem.lemmatize(word.lower()) for word in doc.split()
                                     if letter_only(word) and word not in names]))
    return cleaned_docs

print(labels)
cleaned_mails = clean_txt(mails)

#removing stop words, and extracting features, which are the term frequencies from the cleaned text data
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english", max_features=500)
term_docs = cv.fit_transform(cleaned_mails)

#We can see what the corresponding terms

feature_names = cv.get_feature_names()
#print('term_doc =\n',term_docs[0])
print('feature_names=', feature_names)

#Or by the vocabulary dictionary with term feature as the key and feature index as the value
feature_mapping = cv.vocabulary_
print('feature_mapping = ', feature_mapping)

#group the data by label:
def get_label_index(labels):

    from collections import defaultdict
    label_index = defaultdict(list)
    for index,label in enumerate(labels):
            label_index[label].append(index)
    return label_index

label_index = get_label_index(labels)
print('label index = ', label_index)

#Starting with the prior
def get_prior(label_index):

    prior = {label: len(index) for label, index in label_index.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior

prior = get_prior(label_index)
print ('proir = ',prior)

#now we coding likelihood
import numpy as np

def get_likelihood(term_document_matrix, label_index, smoothing=1):
    likelihood = {}
    for label, index in label_index.items():
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood

likelihood = get_likelihood(term_docs, label_index)
print('len likelihood = ',len(likelihood[0]))
#print(likelihood)
print(feature_names[:5])

#last step coding the posterior
def get_posterior(term_document_matrix, prior,likelihood):
    num_docs = term_document_matrix.shape[0]
    posteriors = []
    for i in range(num_docs):
        posterior = {key: np.log(prior_label) for key, prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            term_document_vector = term_document_matrix.getrow(i)
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index]) * count
        min_log_posterior = min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log_posterior)
            except:
                posterior[label] = float('inf')
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf') :
                posterior[label]= 1.0
            else :
                posterior[label] /= sum_posterior

        posteriors.append(posterior.copy())
    return posteriors

#test the model :
mails_test1 = [
                  '''Subject: flat screens
                  hello ,
                  please call or contact regarding the other flat screens requested .
                  trisha tlapek - eb 3132 b
                  michael sergeev - eb 3132 a
                  also the sun blocker that was taken away from eb 3131 a .
                  trisha should two monitors also michael .
                  thanks
                  kevin moore''' ,
                  '''Subject: re : patchs work better then pillz
worlds first dermal p ; atch technology for p * nis enlarg ; ment
a ; dd 3 + in ; ches today - loo % doc ; tor approved
the viriiity p ; atch r . x . was designed _ for men like yourself who want a b ; lgger , th ; icker , m ; ore en ; ergetic p * nis ! imagine sky _ rocketing in size 2 ' ' , 3 ' ' , even 4 ' ' in 60 _ days or l ; ess . but that ' s not _ all . viriiity p ; atch r . x .
will also super _ charge your s * xual battery effort ; lessly 24 / 7 . your libido and energy level will soar , and you will sat ; isfy your lov ; er like never _ before ! loo % p ; roven to _ work or your m ; oney bac ; k !
to _ get off our listr ; ight here .
i will not spank othersi will not spank othersim 99442 m 6 bb 2 y 384 gosj
36 t 39 qq 5 nfj 55 qjl 2 w 2 o 822 alr 6 y 96 gigccb 4 i 99045 e 5 i will not spank othersxl 6 oolz 9 g 90 oj 218 lj 831 jk
r 2 r 9 vym 311 h 32 ini will not burp in classnu 2 m 442 m 6 bb 2 y 384 gosj 36 t 39 qq 5 n
fj 55 qjl 2 wi will not spank others 2 o 822 alr 6 y 96 gigccb 4 i 99045 e 5 xl 6 o
01 z 9 g 90 oj 218 lj 8 i will not burp in class 31 jkr 2 r 9 vym 311 h 32 innu 2 m 44
2 m 6 bb 2 y 384 gosj 36 t 39 qq 5 nfj 55 qjl 2 w 2 o 8 i will not burp in class
22 alr 6 y 96 gigccb 4 i 990 i will not spank others
45 e 5 xl 6 oolz 9 g 90 oj 218 lj 831 jkr 2 ri will not burp in class i will not spank others
'''
               ]

cleaned_test1 = clean_txt(mails_test1)
term_docs_test = cv.transform(cleaned_test1)
posterior = get_posterior(term_docs_test , prior , likelihood)
print(posterior[0])
print(posterior[1])

#lets do it using sklearn librery

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score , recall_score , f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

X_train, X_test, Y_train, Y_test = train_test_split(cleaned_mails,labels, test_size=0.33, random_state=42)
term_docs_train = cv.fit_transform(X_train)
label_index_train = get_label_index(Y_train)
prior1 = get_prior(label_index_train)
likelihood1 = get_likelihood(term_docs_train , prior1 )
term_docs_test2 = cv.transform(X_test)
posterior1 = get_posterior(term_docs_test2 , prior1 , likelihood1)
#print(posterior1)

clf = MultinomialNB(alpha=1 , fit_prior= True)
clf.fit(term_docs_train,Y_train)
prediction_prob = clf.predict_proba(term_docs_test2)
#print(prediction_prob[0:2])

prediction = clf.predict(term_docs_test2)
print(prediction[0:2])

accuracy = clf.score(term_docs_test2, Y_test)
print('The accuracy using MultinomialNB is: {0:.1f}%'.format(accuracy*100))

cm = confusion_matrix(Y_test , prediction , labels=[0,1])
print(cm)  ## fail to detecet spam mails

ps = precision_score(Y_test , prediction , pos_label=1)
rs =recall_score(Y_test,prediction, pos_label=1)
fs = f1_score(Y_test , prediction , pos_label=1)
fs2 = f1_score(Y_test , prediction , pos_label=0)
#print(ps)
#print(rs)
#print(fs)
#print(fs2)

report = classification_report(Y_test , prediction)
print(report)     #Where avg is the weighted average according to the proportions of classes.

pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.2, 0.1)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
        # if truth and prediction are both 1
            if y == 1:
                true_pos[i] += 1
        # if truth is 0 while prediction is 1
            else:
                false_pos[i] += 1
        else:
            break
#Then calculate the true and false positive rates for all threshold settings (remember there are 516 positive testing samples and 1191 negative ones):
true_pos_rate = [tp / 516.0 for tp in true_pos]
false_pos_rate = [fp / 1191.0 for fp in false_pos]
#Now we can plot the ROC curve with matplotlib:
plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color='orange',lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print(roc_auc_score(Y_test, pos_prob))

k = 10
k_fold = StratifiedKFold(n_splits=k)
max_features_option = [2000, 4000, 8000]
cleaned_mails_np = np.array(cleaned_mails)
smoothing_factor_option = [0.5, 1.0, 1.5, 2.0]
labels_np = np.array(labels)
fit_prior_option=[False,True]
auc_record = {}
for train_indices, test_indices in k_fold.split(cleaned_mails, labels):
    X_train, X_test = cleaned_mails_np[train_indices],cleaned_mails_np[test_indices]
    Y_train, Y_test = labels_np[train_indices],labels_np[test_indices]
    for max_features in max_features_option:
        if max_features not in auc_record:
            auc_record[max_features] = {}
        cv = CountVectorizer(stop_words="english",max_features=max_features)
        term_docs_train = cv.fit_transform(X_train)
        term_docs_test = cv.transform(X_test)
        for smoothing in smoothing_factor_option:
            if smoothing not in auc_record[max_features]:
                auc_record[max_features][smoothing] = {}
            for fit_prior in fit_prior_option:
                clf = MultinomialNB(alpha=smoothing,fit_prior=fit_prior)
                clf.fit(term_docs_train, Y_train)
                prediction_prob = clf.predict_proba(term_docs_test)
                pos_prob = prediction_prob[:, 1]
                auc = roc_auc_score(Y_test, pos_prob)
                auc_record[max_features][smoothing][fit_prior] = auc + auc_record[max_features][smoothing].get(fit_prior, 0.0)


print('max features  smoothing   fit prior    auc'.format(max_features, smoothing, fit_prior, auc/k))
for max_features, max_feature_record in auc_record.items():
    for smoothing, smoothing_record in max_feature_record.items():
        for fit_prior , auc  in smoothing_record.items():
            print('       {0}      {1}      {2}     {3:.4f}'.format(max_features, smoothing, fit_prior, auc/k))

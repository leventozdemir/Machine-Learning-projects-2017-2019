import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import timeit
from sklearn.metrics import accuracy_score

#read a csv file and return the X_Training sampels as type dictionary
train = 'train.csv'
def read_ad_click(train, n, offset = 0):
    X_dict, Y = [], []
    with open(train, 'r')as csvfile:
        reader = csv.DictReader(csvfile)
        for i in range(offset):
            reader.next()
        i = 0
        for row in reader:
            i += 1
            Y.append(int(row['click']))
            del row['click'], row['id'], row['hour'], row['device_id'], row['device_ip']
            #exclude the id, hour, and device_id, device_ip from features
            X_dict.append(row)
            if i >= n:
                break

    return X_dict, Y

n_max = 1000
X_dict_train , Y_train = read_ad_click(train, n_max)
print(X_dict_train[0])

#we transform these dictionary objects (feature: value) into one-hot encoded vectors using DictVectorizer
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_train = dict_one_hot_encoder.fit_transform(X_dict_train)
#print(len(X_train[0])) =5725
#this means we transformed the original 19-dimension categorical features into 5725-dimension binary features.

#we preparing the test data note: i used the training data as testing data because we just here to explane how it works 
X_test_dict , Y_test = read_ad_click(train , n_max)
X_test= dict_one_hot_encoder.transform(X_test_dict)
#print(len(X_test[0]))

#Measure the time it takes to find the best parameters
start_time = timeit.default_timer()

#creat the parameters an give each key more than 1 value
parameters ={'max_depth': [3, 10, None]}

#prepare the classifier
decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)

#use grid search to start searching for the best value for each key in the parameters dictionary
grid_search = GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train, Y_train)
print("--- %0.3fs seconds ---" %(timeit.default_timer() - start_time))
print(grid_search.best_params_)

#Use the model with the optimal parameter to predict unseen cases
decision_tree_best = grid_search.best_estimator_
pos_proba = decision_tree_best.predict_proba(X_test)[:, 1]
prediction = grid_search.predict(X_test)
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(Y_test, pos_proba)))
accuracy = accuracy_score(Y_train, prediction)
print(accuracy*100)
#from sklearn.ensemble import RandomForestClassifier
#if we use the randome forest classifaire best 'mac_depth' = None  and the ROC AUC = 0.724



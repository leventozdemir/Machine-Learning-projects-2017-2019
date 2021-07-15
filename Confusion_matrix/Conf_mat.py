#Don't forget to read Readme file to under stand the data
#import the important libreries to prepare the data
import pandas as pd
import numpy as np

data_path1 = 'adult+stretch.csv'
data_path2 = 'adult-stretch.csv'
data_path3 = 'yellow-small.csv'
data_path4 = 'yellow-small+adult-stretch.csv'

feature_names =['Color','Size', 'Act', 'Age', 'inflated']

data1 = pd.read_csv(data_path1, names=feature_names)
data2 = pd.read_csv(data_path2, names=feature_names)
data3 = pd.read_csv(data_path3, names=feature_names)
data4 = pd.read_csv(data_path4, names=feature_names)

#put all the data togather
color = []
size = []
act = []
age =[]
inflated = []

#append the list and convet the str to int
#yellow = 0,purple = 1
for i in data1['Color']:
    if i =='YELLOW':
        i=0
    else :
        i = 1
    color.append(i)
for i in data2['Color']:
    if i =='YELLOW':
        i=0
    else :
        i = 1
    color.append(i)
for i in data3['Color']:
    if i =='YELLOW':
        i=0
    else :
        i = 1
    color.append(i)
for i in data4['Color']:
    if i =='YELLOW':
        i=0
    else :
        i = 1
    color.append(i)

#large = 0,small = 1
for i in data1['Size']:
    if i =='LARGE':
        i=0
    else:
        i=1
    size.append(i)
for i in data2['Size']:
    if i =='LARGE':
        i=0
    else:
        i=1
    size.append(i)
for i in data3['Size']:
    if i =='LARGE':
        i=0
    else:
        i=1
    size.append(i)
for i in data4['Size']:
    if i =='LARGE':
        i=0
    else:
        i=1
    size.append(i)

#stretch = 0,dip = 1
for i in data1['Act']:
    if i =='STRETCH':
        i=0
    else:
        i=1
    act.append(i)
for i in data2['Act']:
    if i =='STRETCH':
        i=0
    else:
        i=1
    act.append(i)
for i in data3['Act']:
    if i =='STRETCH':
        i=0
    else:
        i=1
    act.append(i)
for i in data4['Act']:
    if i =='STRETCH':
        i=0
    else:
        i=1
    act.append(i)

#adult=0 ,child=1
for i in data1['Age']:
    if i =='ADULT':
        i=0
    else:
        i=1
    age.append(i)
for i in data2['Age']:
    if i =='ADULT':
        i=0
    else:
        i=1
    age.append(i)
for i in data3['Age']:
    if i =='ADULT':
        i=0
    else:
        i=1
    age.append(i)
for i in data4['Age']:
    if i =='ADULT':
        i=0
    else:
        i=1
    age.append(i)

#false=0, true=1
for i in data1['inflated']:
    if i =='F':
        i=0
    else:
        i=1
    inflated.append(i)
for i in data2['inflated']:
    if i =='F':
        i=0
    else:
        i=1
    inflated.append(i)
for i in data3['inflated']:
    if i =='F':
        i=0
    else:
        i=1
    inflated.append(i)
for i in data4['inflated']:
    if i =='F':
        i=0
    else:
        i=1
    inflated.append(i)

color = np.array(color)
size = np.array(size)
act = np.array(act)
age =np.array(age)
inflated = np.array(inflated)

#put all the lists togather
data_array = np.concatenate((color,size,act,age,inflated)).reshape((76,5))
#print(data_array)

#make the data frame
data = pd.DataFrame(data_array, columns=feature_names)
#print(data)

#preprare x and y arrays for training
Y_array = data['inflated']
#print(Y_array)
X_array = data.drop(['inflated'],axis=1)
print(X_array)

#split the data and train the model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_array,Y_array,test_size=0.05,random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
clf = DecisionTreeClassifier(min_samples_split=2)
clf.fit(X_train,Y_train)
#make a prediciton to evaluate the model
predict =clf.predict(X_test)

conf_mat =confusion_matrix(Y_test, predict)
print(conf_mat)
#the confusion matrix of the model is :[[2 0]
#                                       [1 1]]
#precisionT(1) = T1/(T1+F1) = 1
#precisionF(0) = T0/(T0+F0) = 2/(2+1) = 0.6666
#recallF(0)(sensitivity) = T0/(T0+F1)=2/2=1
#recallT(1)(Specificaiti) = T1/(T1+F0)=1/(1+1)=0.5
#F1_scoreT = 2*(preT*recallT)/(preT+recallT)=2*(0.5)/(1.5)=0.66666
#F1_scoreF = 2*(preF*recallF)/(preF+recallF)=2*(0.67)/(1.67)=0.8023
#accuracy = (T0+T1)/(T0+T1+F0+F1)=3/(2+1+1+0)=3/4=0.75

#we can see it in the classification report
print(classification_report(Y_test, predict))
print("the accuracy of the classifier:",accuracy_score(Y_test, predict)*100)


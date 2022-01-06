import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
#les imports el kol


heart= pd.read_csv('heart.csv')                 #importation du data
X=heart.drop(columns='target',axis=1)           #features houni axis=1 khter houni bch nedropiw colonne
Y=heart['target']                               #7atina target houni ya3ni labels 

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size=0.2) # zedna kasamna data mte3na l train w test ( x_train  x_test w y_train w y_test)

naive = GaussianNB()
naive.fit(X_Train, Y_Train)                        #3malna training al model mte3na 


X_Train_pred=naive.predict(X_Train)
train_acc = accuracy_score(X_Train_pred, Y_Train) #kharajna accuracy_score houni



y_preds=naive.predict(X_Test)


gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()

gbc.fit(X_Train, Y_Train)
rfc.fit(X_Train, Y_Train)




data=pd.DataFrame(X_Test['cp'])
data['target']=y_preds
data.to_csv("subbmit.csv", index=False)  #houni kharajna fichier csv samineh subbmit meli khdemneh kbal bch nchoufou selon 'chest pain type' la personne est malade ou pas


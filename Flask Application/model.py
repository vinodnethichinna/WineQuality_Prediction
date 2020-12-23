#Importing required libraries.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV



import numpy as np
import pickle
import xgboost

#Reading the input data.
data = pd.read_csv('archive\winequality-white.csv')

# changing columns names and making it standard.
data.rename(columns = {
                            'fixed acidity' : 'fixed_acidity',
                            'volatile acidity' : 'volatile_acidity',
                            'citric acid' : 'citric_acid',                            
                            'residual sugar'  : 'residual_sugar',
                            'free sulfur dioxide' : 'free_sulfur_dioxide',
                            'total sulfur dioxide' : 'total_sulfur_dioxide'
                          },inplace = True)

data.drop(data.loc[data['fixed_acidity'] >14].index,inplace = True)
data.drop(data.loc[data['residual_sugar'] >40].index,inplace = True)

X = data.drop(columns = ['quality'])
var_threshold = VarianceThreshold(threshold=.3)
var_threshold.fit(X)
X_var_threshold = X.iloc[:,var_threshold.get_support()]

# features and target columns.

features = X_var_threshold
target = data['quality']

#split the transformed data into train and test datasets.

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.25,random_state=42,stratify = target)

# hyper tuning XGB Classifier.

xgb_parameters = {
    "learning_rate" : [0.1,0.2,0.3,0.5,0.10],
    "max_depth" : [3,4,6,7,8,9],
    "min_child_weight" : [1,2,3,4,5,6],
    "gamma" : [0.0,0.1,0.2,0.3,0.4,0.8]
    }  

classifier_model = xgboost.XGBClassifier()
random_search = RandomizedSearchCV(classifier_model,param_distributions = xgb_parameters,cv = 3,verbose = 3)
random_search.fit(X_train,Y_train)

# listing out the best parameters.
random_search.best_params_

# finalizing the model and checking the accuracy again with best parameters.
model_final = XGBClassifier(min_child_weight = 3,
                            max_depth = 8,
                            learning_rate = 0.3,
                            gamma = 0.0,n_estimators=100)

#fitting the model with training data.
model_final.fit(X_train,Y_train)

#predicting test dataset using XGB Classifier.
y_pred = model_final.predict(X_test)

# Saving model using pickle
pickle.dump(model_final, open('model.pkl','wb'))
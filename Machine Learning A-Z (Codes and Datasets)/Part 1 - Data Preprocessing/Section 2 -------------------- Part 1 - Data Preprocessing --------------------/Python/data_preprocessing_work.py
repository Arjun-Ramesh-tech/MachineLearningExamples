#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:35:04 2020

@author: arjrames
"""

#Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
#Importing DataSet
dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#Preprocessing

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
X[:,1:3] = imputer.fit_transform(X[:,1:3])

#One Hot Encoding

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

#LabelEncoder
le=LabelEncoder()
Y=le.fit_transform(Y)

#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)

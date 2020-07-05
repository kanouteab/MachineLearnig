#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:40:34 2019

@author: macbook
"""
#Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importer les datasets

dataset=pd.read_csv('Data.csv')
x_norm = dataset.iloc[:, :-1].values
y_norm = dataset.iloc[:, -1].values
x=pd.DataFrame(x_norm)
y=pd.DataFrame(y_norm)
x1 = dataset.iloc[:, :-1].values
#Gerer les données manquantes
from sklearn.preprocessing import Imputer
imputer = Imputer (missing_values = 'NaN',strategy = 'mean', axis = 0)
imputer.fit(x_norm[:,1:3])
x_norm[:,1:3] = imputer.transform(x_norm[:,1:3])
x_value = pd.DataFrame(x_norm)

#Gerer les variables catégoriques
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder ()
x_norm[:,0] = labelencoder_X.fit_transform(x_norm[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x_norm = pd.DataFrame(onehotencoder.fit_transform(x_norm).toarray())
labelencoder_y = LabelEncoder ()
ym = labelencoder_y.fit_transform(y_norm)
#x_hotenc=pd.DataFrame(x_norm)

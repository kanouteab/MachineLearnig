#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 01:20:16 2019

@author: macbook
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3, 4]].values

# Utiliser la méthode elbow pour trouver le nombre optimal de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('La méthode Elbow')
plt.xlabel('Nombre de clusters')
plt.ylabel('WCSS')
plt.show()

#Construction du model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_means = kmeans.fit_predict(x)
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], c = 'red', label = 'clauster 1')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], c = 'blue', label = 'clauster 2')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], c = 'green', label = 'clauster 3')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], c = 'black', label = 'clauster 4')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], c = 'yellow', label = 'clauster 5')
plt.title('Clauster de client')
plt.xlabel('Salaire annuel')
plt.ylabel('Spending score')
plt.legend()
"""
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_means = kmeans.fit_predict(x)
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], c = 'red', label = 'clauster 1')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], c = 'blue', label = 'clauster 2')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], c = 'green', label = 'clauster 3')
plt.title('Clauster de client')
plt.xlabel('Salaire annuel')
plt.ylabel('Spending score')
plt.legend()
"""
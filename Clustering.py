# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

data = pd.read_csv('cs1-scores.txt',delimiter='\t',comment='#')
features = data.columns
data=pd.DataFrame(normalize(data, norm='max', axis=0), columns=features)
data['Assignments']=data.mean(axis=1)
kmeans = KMeans(n_clusters=7)
kmeans.fit(data[['Final','Assignments']])
plt.figure(1,figsize=(10,10))
plt.scatter(data['Final'],data['Assignments'])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='x')
plt.xlabel('Final Exam Score')
plt.ylabel('Mean Assignment Score')
plt.title('Clustering')
plt.savefig('Clustering.png')
plt.close
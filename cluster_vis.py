# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:17:18 2023

@author: smahmud
"""

from yellowbrick.cluster import SilhouetteVisualizer
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

#
# Load  dataset
#
df = pd.read_csv(
    r"\\kcc-mdstore01\Public\SCS\ML_Derecho\BlobFiles\NEXRAD\objects\19960101_RadarPolygonPoints55_100_BinaryDilatation2Erosion.csv")

x = df.RadarPolygonPoints_X
y = df.RadarPolygonPoints_Y
x = x.to_numpy()

y = y.to_numpy()
xy = np.transpose((x, y))
X = xy
#
# Instantiate the KMeans models
#
# %%

# range_n_clusters = [2, 3, 4, 5, 6]
# for n_clusters in range_n_clusters:
#     clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
#     cluster_labels = clusterer.fit_predict(X)

#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print(
#         "For n_clusters =",
#         n_clusters,
#         "The average silhouette_score is :",
#         silhouette_avg,
#     )

km = KMeans(n_clusters=3, random_state=42)
# Fit the KMeans model

km.fit_predict(X)

# Calculate Silhoutte Score

score = silhouette_score(X, km.labels_, metric='euclidean')

# Print the score

print('Silhouetter Score: %.3f' % score)

# %%


fig, ax = plt.subplots(2, 2, figsize=(15, 8))
for i in [2, 3, 4, 5]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++',
                n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(
        km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X)

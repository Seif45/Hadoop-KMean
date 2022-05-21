from sklearn.cluster import KMeans
import pandas as pd
import time

startTime = time.time()
columns = ['1','2','3','4','5']
irisdata = pd.read_csv('iris.data', names=columns)
irisdata['5'] = pd.Categorical(irisdata["5"])
irisdata["5"] = irisdata["5"].cat.codes
X = irisdata.values[:, 0:4]
y = irisdata.values[:, 4]
# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_
print(kmeans.n_iter_)
print(centroids)
print((time.time() - startTime) * 1000 , " ms")

from sklearn.cluster import KMeans
import numpy as np
import os
import matplotlib.pyplot as plt
import fastcluster
import scipy.cluster.hierarchy as sch
from sklearn.cluster import SpectralClustering
import argparse 
from sklearn.cluster import KMeans
from pathlib import Path

# for no. of clusters self find https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/
def hierarchy_clustering(embeddings):
	linkage_matrix = fastcluster.linkage(embeddings, method='ward')
	# Cut the linkage matrix to obtain the cluster labels
	dendrogram = sch.dendrogram(linkage_matrix)
	# Decide the number of clusters based on the dendrogram
	num_clusters = len(set(dendrogram['color_list']))
	# Get the cluster labels
	labels = sch.fcluster(linkage_matrix, num_clusters, criterion='maxclust')
	# Print the labels
	return labels
def kmeans_clustering_plot(data): #elbow method is commented
	# create an instance of the KMeans model with k clusters
	model = KMeans(n_clusters=2)
	# fit the model to the data and predict the labels of the data points
	labels = model.fit_predict(data)
	#inertia = model.inertia_
	# loop through different values of k
	#k_values = range(1, 4)
	#inertias = []
	#for k in k_values:
	#   model = KMeans(n_clusters=k)
	#   model.fit(data)
	#   inertias.append(model.inertia_)
	# plot the data points and their labels
	#plt.figure(figsize=(10, 5))
	#plt.plot(range(len(labels)), labels)
	#plt.show()
	#plt.plot(k_values, inertias)
	#plt.show()
	return labels
def spectral_clustering(data):
	embeddings =data 
	sc = SpectralClustering(n_clusters=2, random_state=0).fit(embeddings)
	clusters = sc.fit_predict(embeddings)
	return clusters
def kmeans_self_no_of_clusters(data):
	embeddings = data
	kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
	labels = kmeans.fit_predict(embeddings)
	return labels	


parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
target_file = Path(args.path)

if not target_file.exists():
    print("The  target_file doesn't exist")
    raise SystemExit(1)

data=np.load(target_file)
hc=hierarchy_clustering(data)
#kmeans_2_clusters=kmeans_clustering_plot(data)#no. of clusters provided
kmeans_self=kmeans_self_no_of_clusters(data)
sc=spectral_clustering(data)
stacked_lists = np.stack([hc,kmeans_self,sc])
f_name=args.path
if ".npy" in f_name:
	f_name=f_name.split(".npy")[0]

np.save('stacked_outputs_'+f_name+'.npy', stacked_lists)



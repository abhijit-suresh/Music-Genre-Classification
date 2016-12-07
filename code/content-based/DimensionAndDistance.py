#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.decomposition import PCA, FastICA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import GaussianRandomProjection
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from emd import emd

import os

tracks = []
for subdir, dirs, files in os.walk('./tracks/'):
    for file in files:
      if file.endswith('.wav'):
      	tracks.append(str(file))

size = len(tracks)
pca_result = [];

def reduceDimension(wav_file,index):
	(rate,sig) = wav.read('./tracks/'+wav_file)
	mfcc_feat = mfcc(sig,rate,numcep=15)
	# result = []
	# pca = PCA(n_components=13)
	# temp = pca.fit_transform(mfcc_feat)
	# for item in range(0,temp.shape[1],1):
	# 	result.append(np.mean(temp[:,item]))
	# result = np.array(result)
	# result = []
	# ica = FastICA(n_components=13)
	# temp = ica.fit_transform(mfcc_feat)
	# for item in range(0,temp.shape[1],1):
	# 	result.append(np.mean(temp[:,item]))
	# result = np.array(result)
	result = []
	rp = GaussianRandomProjection(n_components=13)
	temp = rp.fit_transform(mfcc_feat)
	for item in range(0,temp.shape[1],1):
		result.append(np.mean(temp[:,item]))
	result = np.array(result)

	pca_result.append(result);

print 'Reducing Dimension....'
for i in range(0,size,1):
	reduceDimension(tracks[i],i)
print 'Completed Dimension reduction!'

sim_result = [];

def calculateDistance(pca_arr):
	for i in range(0,size,1):
		temp = []
		a = pca_arr[i]
		for j in range(0,size,1):
			b = pca_arr[j]
			dist = LA.norm(a-b)
			# dist = emd(a,b)
			print dist
			temp.append(dist)
		# print i,np.array(temp).shape
		sim_result.append(temp)

def mydist(a,b):
	return a+b
print 'Computing Similarity distance'
calculateDistance(pca_result)
print 'Similarity distance calculated!'

# X = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]
# neigh = KNeighborsClassifier(n_neighbors=3,algorithm='auto', metric=lambda a,b: mydist(a,b))
# neigh.fit(X, y)
# print(neigh.predict([[1.1]]))
im = plt.imshow(sim_result, extent=[0,729,0,729], aspect='auto')
plt.title('Distance Matrix with Random Projection')
plt.xlabel('Track Index')
plt.colorbar(im, orientation='vertical')
plt.show()


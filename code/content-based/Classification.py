#!/usr/bin/env python

from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
from emd import emd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn import manifold
import math
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

genre = {'classical':1,'electronic':2,'jazz_blues':3,'metal_punk':4,'rock_pop':5,'world':6}
ground_data = []

with open('ground_truth.txt') as f:
	for line in f.readlines():
		data = line.split('\t')
		data[1] =  genre.get(data[1].split('\r\n')[0],'None')
		ground_data.append(data)

signature = []

with open('sign_result_knn.txt') as f:
	for line in f.readlines():
		data = line.split('\t')
		temp = []
		for item in data:
			if item != '\n':
				temp.append(item)
		temp = np.array(temp)
		temp = temp.astype(float)
		temp = np.reshape(temp,(10,3))
		signature.append(temp)

matrix = [[0 for x in range(729)] for y in range(729)]

for i in range(0,729,1):
	temp1 =  signature[i]
	for j in range(0,729,1):
		temp2 = signature[j]
		# temp3 = entropy(pk, qk=None,
		dist = emd(temp1,temp2)
		matrix[i][j] = dist

im = plt.imshow(matrix, extent=[0,729,0,729], aspect='auto')
plt.title('Distance Matrix')
plt.xlabel('Track Index')
plt.colorbar(im, orientation='vertical')
nmds = manifold.MDS(n_components=2, metric=False, dissimilarity="precomputed")
npos = nmds.fit_transform(np.array(matrix))

classical = []
electronic = []
jazz_blues = []
metal_punk = []
rock_pop = []
world = []
for i in range(0,729,1):
	if ground_data[i][1] == 1:
		classical.append(npos[i])
	elif ground_data[i][1] == 2:
		electronic.append(npos[i])
	elif ground_data[i][1] == 3:
		jazz_blues.append(npos[i])
	elif ground_data[i][1] == 4:
		metal_punk.append(npos[i])
	elif ground_data[i][1] == 5:
		rock_pop.append(npos[i])
	elif ground_data[i][1] == 6:
		world.append(npos[i])


classical = np.array(classical)
electronic = np.array(electronic)
jazz_blues = np.array(jazz_blues)
metal_punk = np.array(metal_punk)
rock_pop = np.array(rock_pop)
world = np.array(world)

cross_count = math.floor(0.7 * len(ground_data))
i_cross = random.sample(range(0, 728), int(cross_count))

train_data = []
test_data = []

for i in range(0,729,1):
	if i in i_cross:
		train_data.append([i,ground_data[i][1]])
	else:
		test_data.append([i,ground_data[i][1]])

X = []
y = []
for item in train_data:
	index = item[0]
	y.append(item[1])
	X.append(npos[index])

# final_scores = []
f = open('results.txt','w')
clf = KNeighborsClassifier(n_neighbors=20,algorithm='auto')
# clf = cluster.SpectralClustering(n_clusters=i,eigen_solver='arpack', affinity="nearest_neighbors")
clf.fit(X, y)

# clf =  SVC()
# clf.fit(X, y)

# clf = LinearDiscriminantAnalysis()
# clf.fit(X, y)

t = []
for item in test_data:
	index = item[0]
	t.append(npos[index])

score = 0
total_test_count = len(test_data)
for i in range(0,total_test_count,1):
	f.write(str(clf.predict(t[i])[0]))
	f.write(str(test_data[i][1]))
	f.write('\n')
	if clf.predict(t[i])[0] == test_data[i][1]:
		score += 1

print 'Validation accuracy: ',(float(score) * 100)/float(total_test_count),' %'
	# final_scores.append((float(score) * 100)/float(total_test_count))
f.close()
# plt.plot(xrange(1,30),final_scores)
# plt.show()
# colors = ['b', 'c', 'y', 'm', 'r']

# lo = plt.scatter(classical[:,0],classical[:,1],marker='x', color=colors[0])
# ll = plt.scatter(electronic[:,0],electronic[:,1],marker='o', color=colors[1])
# l = plt.scatter(jazz_blues[:,0],jazz_blues[:,1],marker='x', color=colors[2])
# a = plt.scatter(metal_punk[:,0],metal_punk[:,1],marker='o', color=colors[3])
# h = plt.scatter(rock_pop[:,0],rock_pop[:,1],marker='x', color=colors[3])
# hh = plt.scatter(world[:,0],world[:,1],marker='o', color=colors[4])
# plt.legend((lo, ll, l, a, h, hh),
#            ('classical', 'electronic', 'jazz_blues', 'rock_pop', 'metal_punk', 'world'),
#            scatterpoints=1,
#            loc='lower left',
#            ncol=3,
#            fontsize=8)
# plt.show()
#List of Genres
# classical - 1
# electronic - 2
# jazz_blues - 3
# metal_punk - 4
# rock_pop - 5
# world - 6

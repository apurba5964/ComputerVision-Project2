import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


k=3

x_points = [5.9,4.6,6.2,4.7,5.5,5.0,4.9,6.7,5.1,6.0]
y_points = [3.2,2.9,2.8,3.2,4.2,3.0,3.1,3.1,3.8,3.0]


coordinates = np.array(list(zip(x_points, y_points)))
#print(X)

#(6.2, 3.2) (red), µ2 = (6.6, 3.7) (green), µ3 = (6.5, 3.0) (blue)
c_x=[6.2,6.6,6.5]
c_y=[3.2,3.7,3.0]

centeroids = np.array(list(zip(c_x, c_y)), dtype=np.float32)

centeroids_old = np.zeros(centeroids.shape)
#print(C)
clusters = np.zeros(len(coordinates))
colors = ['r', 'g', 'b']


def calculateDistance(a, b, ax=1):
	return np.linalg.norm(a - b, axis=ax)

for iter in range(1,3):


	plt.scatter(x_points, y_points, s=80, facecolors='none', edgecolors='b', marker='^')
	plt.scatter(centeroids[:, 0], centeroids[:, 1], color=['red', 'green', 'blue'], marker='o', lw=2)

	for i,j in zip(x_points, y_points):
		plt.annotate('%s)' %j, xy=(i,j), xytext=(30,0), textcoords='offset points')
		plt.annotate('(%s,' %i, xy=(i,j))


	for i,j in zip(centeroids[:, 0], centeroids[:, 1]):
		plt.annotate('%s)' %j, xy=(i,j), xytext=(30,0), textcoords='offset points')
		plt.annotate('(%s,' %i, xy=(i,j))
	
	if(iter==1):
		plt.savefig('task3_iter1_a.jpg', transparent=True, bbox_inches='tight', pad_inches=0)
	if(iter==2):
		plt.savefig('task3_iter2_a.jpg', transparent=True, bbox_inches='tight', pad_inches=0)


	for i in range(len(coordinates)):

		distances = calculateDistance(coordinates[i], centeroids)
		cluster = np.argmin(distances)
		clusters[i] = cluster
		print(clusters)


	for i in range(k):
		points = np.array([coordinates[j] for j in range(len(coordinates)) if clusters[j] == i])

		plt.scatter(points[:, 0], points[:, 1], s=80, facecolors=colors[i],marker='^')


	if(iter==1):
		plt.savefig('task3_iter1_b.jpg', transparent=True, bbox_inches='tight', pad_inches=0)
	if(iter==2):
		plt.savefig('task3_iter2_b.jpg', transparent=True, bbox_inches='tight', pad_inches=0)	


	for i in range(k):
	    points = np.array([coordinates[j] for j in range(len(coordinates)) if clusters[j] == i])
	    centeroids[i]=np.around(np.mean(points,axis=0),decimals=2)

	print(centeroids)
	plt.clf()


#print('-------------2nd Iteration---------------')

#plt.scatter(centeroids[:, 0], centeroids[:, 1], color=['red', 'green', 'blue'], marker='o', lw=2)
#plt.show()
'''
for i in range(len(coordinates)):

    distances = calculateDistance(coordinates[i], centeroids)
    #print(distances)
    cluster = np.argmin(distances)
    #print(cluster)
    clusters[i] = cluster
    print(clusters)

for i in range(k):
	points = np.array([coordinates[j] for j in range(len(coordinates)) if clusters[j] == i])
	#print(points)
	plt.scatter(points[:, 0], points[:, 1], s=80, facecolors=colors[i],marker='^')


plt.savefig('task3_iter1_a.jpg', transparent=True, bbox_inches='tight', pad_inches=0)

'''

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
import random
UBIT = 'APURBAMA'; 
np.random.seed(sum([ord(c) for c in UBIT]))


#k=3
k=3

img = cv.imread('baboon.jpg')
#print(img.shape)


color = np.random.randint(0,255, size=(k, 3)).tolist()



centeroids = np.array(list(color), dtype=np.float32)
print(centeroids)


coordinates = np.array(list(img))
coordinates_bkp = np.array(list(img))


def calculateDistance(a, b, ax=1):
	return np.linalg.norm(a - b, axis=ax)


for i in range(10):
	x =np.zeros((k,3),dtype=float)
	xr=[]
	xg=[]
	xb=[]
	yr=[]
	yg=[]
	yb=[]
	zr=[]
	zg=[]
	zb=[]
	coordinates = np.array(list(img))


	clusters = np.zeros(coordinates.shape)




	for i in range(len(coordinates)):
	    for j in range(len(coordinates)):
		    distances = calculateDistance(coordinates[i][j], centeroids)

		    cluster = np.argmin(distances)
		    #print(cluster)
		    clusters[i][j]=cluster
		    # print(clusters.shape)
		    if(cluster==0):
			    #print(coordinates[i][j].shape)
			    xr.append((coordinates[i][j])[0])
			    yr.append((coordinates[i][j])[1])
			    zr.append((coordinates[i][j])[2])

			    coordinates[i][j]=centeroids[0]
			#coord0i_copy.append(i)
			#coord0j_copy.append(j)
		    if(cluster==1):
			#y[i][j] = coordinates[i][j]
			    xg.append((coordinates[i][j])[0])
			    yg.append((coordinates[i][j])[1])
			    zg.append((coordinates[i][j])[2])
			    coordinates[i][j]=centeroids[1]
			#coord1i_copy.append(i)
			#coord1j_copy.append(j)
		    if(cluster==2):
			#z[i][j] = coordinates[i][j]
			    xb.append((coordinates[i][j])[0])
			    yb.append((coordinates[i][j])[1])
			    zb.append((coordinates[i][j])[2])
			    coordinates[i][j]=centeroids[2]

	centeroids[0][0]=np.mean(xr)
	centeroids[0][1]=np.mean(yr)
	centeroids[0][2]=np.mean(zr)
	centeroids[1][0]=np.mean(xg)
	centeroids[1][1]=np.mean(yg)
	centeroids[1][2]=np.mean(zg)
	centeroids[2][0]=np.mean(xb)
	centeroids[2][1]=np.mean(yb)
	centeroids[2][2]=np.mean(zb)


    #print(centeroids)
#print(np.mean(x,axis=0).shape)
#print(np.mean(y,axis=0).shape)
#print(np.mean(z,axis=0).shape)
	
#for i in range(0,3):
#	centeroids[i][i]=np.mean(x,axis=0)


			

#print(clusters[2][1])
cv.imwrite('task3_baboon_3.jpg',coordinates)

'''
for i in range(k):
	for j in range(k):

		points = np.array([coordinates[i][j] for a in range(len(coordinates)) if clusters[][j] == i])		
'''


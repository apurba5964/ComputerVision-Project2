import numpy as np
import cv2 as cv
import random
UBIT = 'APURBAMA'; 
np.random.seed(sum([ord(c) for c in UBIT]))



img1 = cv.imread('tsucuba_left.png',0) #train image
img2 = cv.imread('tsucuba_right.png',0) #test image

def genKeypointTask1():
	img = cv.imread('tsucuba_left.png')
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	sift = cv.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray,None)
	img=cv.drawKeypoints(gray,kp,img)
	cv.imwrite('task2_sift1.jpg',img)

def genKeypointTask2():

    img2 = cv.imread('tsucuba_right.png')
    gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    sift2 = cv.xfeatures2d.SIFT_create()
    kp2, des2 = sift2.detectAndCompute(gray2,None)
    img2=cv.drawKeypoints(gray2,kp2,img2)
    cv.imwrite('task2_sift2.jpg',img2)

def generateKNN():
	img1 = cv.imread('tsucuba_left.png',0) #train image
	img2 = cv.imread('tsucuba_right.png',0) #test image
	
	#Initialize sift detector
	sift = cv.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	bruteForce = cv.BFMatcher()
	matches = bruteForce.knnMatch(des1,des2, k=2)

	best = []
	best_match1 = []
	best_match2 = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			best.append([m])
			best_match1.append(kp1[m.queryIdx].pt)
			best_match2.append(kp2[m.trainIdx].pt)

	
	img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,best,None, flags=2)

	cv.imwrite('task2_matches_knn.jpg',img3)
	return best_match1,best_match2   


def generateRandomInliers(best_match1,best_match2):
	y=np.zeros((10,2), dtype=int)
	z=np.zeros((10,2), dtype=int)
	#print(best_match1.shape)
	x=np.array(np.random.randint(low=0, high=len(best_match1), size=10))

	for i in range (0,len(x)):
		#print(best_match1[i])
		#print('-----------------------')
		y[i] = best_match1[x[i]]
		z[i] = best_match2[x[i]]
		#print(y)
		
		#print(z)

	#print(y)
	#print(z)	

	return y,z







def generateFundamentalMatrix(best_match1,best_match2):
	best_match1 = np.int32(best_match1)
	best_match2 = np.int32(best_match2)
	F, mask = cv.findFundamentalMat(best_match1,best_match2,cv.FM_RANSAC)
	print("-------Fundamental Matrix--------------")
	print(F)
	
	#Selecting only inlier points
	best_match1 = best_match1[mask.ravel()==1]
	best_match2 = best_match2[mask.ravel()==1]


	sample_best_match1,sample_best_match2=generateRandomInliers(best_match1,best_match2)

	#sample_best_match1=best_match1[np.random.choice(best_match1.shape[0], 10, replace=False)]
	#sample_best_match2=best_match2[np.random.choice(best_match2.shape[0], 10, replace=False)]

	#print(np.array(sample_best_match1))
	#print(sample_best_match1.shape)
	#print(best_match2)

	color = np.random.randint(0,255, size=(10, 3)).tolist()
	#print(color)

	#Draw lines on the right image
	
	lines2 = cv.computeCorrespondEpilines(sample_best_match1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3,img4 = drawlines(img2,img1,lines2,sample_best_match2,sample_best_match1,color)

    #Draw lines on the left image
	lines1 = cv.computeCorrespondEpilines(sample_best_match2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img5,img6 = drawlines(img1,img2,lines1,sample_best_match1,sample_best_match2,color)

	cv.imwrite('task2_epi_right.jpg',img3)
	cv.imwrite('task2_epi_left.jpg',img5)
	
	



def drawlines(img1,img2,lines,pts1,pts2,color_list):
    
    r,c = img1.shape
    #color = tuple(color)
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2,color in zip(lines,pts1,pts2,color_list):
        color = tuple(color)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def generateDisparity(imgL, imgR):

    window_size = 5
    min_disp = 16
    num_disp = 64-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32)

    
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    disp = np.abs(disp-min_disp)/num_disp

    disp = disp * 300
    #print(disp)
    #cv.imshow('disparity', (disp-min_disp)/num_disp)

    cv.imwrite('task2_disparity.jpg',disp)



genKeypointTask1()
genKeypointTask2()

best_match1,best_match2=generateKNN()
generateFundamentalMatrix(best_match1,best_match2)	 

generateDisparity(img1,img2)

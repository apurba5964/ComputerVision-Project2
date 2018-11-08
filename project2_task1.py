import numpy as np
import cv2 as cv
import random
UBIT = 'APURBAMA'; 
np.random.seed(sum([ord(c) for c in UBIT]))



def genKeypointTask1():
	img = cv.imread('mountain1.jpg')
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	sift = cv.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray,None)
	img=cv.drawKeypoints(gray,kp,img)
	cv.imwrite('task1_sift1.jpg',img)

def genKeypointTask2():

    img2 = cv.imread('mountain2.jpg')
    gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    sift2 = cv.xfeatures2d.SIFT_create()
    kp2, des2 = sift2.detectAndCompute(gray2,None)
    img2=cv.drawKeypoints(gray2,kp2,img2)
    cv.imwrite('task1_sift2.jpg',img2)


def generateKNN():
	img1 = cv.imread('mountain1.jpg',0) #train image
	img2 = cv.imread('mountain2.jpg',0) #test image
	
	#Initialize sift detector
	sift = cv.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	bruteForce = cv.BFMatcher()
	matches = bruteForce.knnMatch(des1,des2, k=2)

	best = []
	best_match = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			best.append([m])
			best_match.append(m)

	
	img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,best,None, flags=2)

	cv.imwrite('task1_matches_knn.jpg',img3)
	return best_match,kp1,kp2

def remove_values_from_list(the_list,best_match):

	x=[]
	y=[]
	y_sample=[]
	
	for i in range(0,len(the_list)):
		if(the_list[i]==1):

		    x.append(the_list[i])
		    y.append(best_match[i])

	randlist=np.random.randint(low=0, high=len(y), size=10)	    

	for j in range(0,10):
		y_sample.append(y[randlist[j]])


	return x,y_sample





def createPanoImage(best_match,kp1,kp2):
	img1RGB = cv.imread('mountain1.jpg') #train image
	img2RGB = cv.imread('mountain2.jpg') #test image
	img1 = cv.imread('mountain1.jpg',0) #train image
	img2 = cv.imread('mountain2.jpg',0) #test image
	
	srcPoints = np.float32([ kp1[m.queryIdx].pt for m in best_match ]).reshape(-1,1,2)
	dstPints = np.float32([ kp2[m.trainIdx].pt for m in best_match ]).reshape(-1,1,2)

	H, mask = cv.findHomography(srcPoints, dstPints, cv.RANSAC, 4.0)
	matchesMask = mask.ravel().tolist()
	inLierMatchesMask,sample_best_match_inlier = remove_values_from_list(matchesMask,best_match)

	sample_inLierMatchesMask=inLierMatchesMask[:10]
	#sample_best_match_inlier=random.sample(best_match_inlier, 10)
	#np.random.randint(low=0, high=len(best_match1), size=10)

	print("-------Homography Matrix--------------")
	print(H)
	#print(len(inLierMatchesMask))
	#print(len(best_match_inlier))
	#generateRandomInliers(inLierMatchesMask,best_match_inlier)
	
	

	draw_params = dict(matchColor = (0,255,0),
	           singlePointColor = None,
	           matchesMask = sample_inLierMatchesMask,
	           flags = 2)

	task1_matches_homo = cv.drawMatches(img1, kp1, img2, kp2, sample_best_match_inlier, None, **draw_params)

	(size, offset) = calculate_size(img1.shape, img2.shape, H)
	pano=merge_images(img1RGB,img2RGB,H,size,offset,(kp1,kp2))


	cv.imwrite('task1_matches.jpg',task1_matches_homo)

	cv.imwrite('task1_pano.jpg',pano)






def merge_images(image1, image2, hmatrix, size, offset, keypoints):

  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  
  panorama = np.zeros((size[1], size[0], 3), np.uint8)
  
  (offsetx, offsety) = offset
  
  translation = np.matrix([
    [1.0, 0.0, offsetx],
    [0, 1.0, offsety],
    [0.0, 0.0, 1.0]
  ])
  
  
  hmatrix = translation * hmatrix
  
  cv.warpPerspective(image1, hmatrix, size, panorama)
  
  panorama[offsety:h1+offsety, offsetx:offsetx+w1] = image2  
  
  return panorama



def calculate_size(size_image1, size_image2, hmatrix):
  
  (h1, w1) = size_image1[:2]
  (h2, w2) = size_image2[:2]
  
  #remapping the coordinates of the projected image onto the panorama image space
  top_left = np.dot(hmatrix,np.asarray([0,0,1]))
  top_right = np.dot(hmatrix,np.asarray([w2,0,1]))
  bottom_left = np.dot(hmatrix,np.asarray([0,h2,1]))
  bottom_right = np.dot(hmatrix,np.asarray([w2,h2,1]))
  
  
  #normalize
  top_left = top_left/top_left[2]
  top_right = top_right/top_right[2]
  bottom_left = bottom_left/bottom_left[2]
  bottom_right = bottom_right/bottom_right[2]

  
  
  pano_left = int(min(top_left[0], bottom_left[0], 0))
  pano_right = int(max(top_right[0], bottom_right[0], w1))
  W = pano_right - pano_left
  
  pano_top = int(min(top_left[1], top_right[1], 0))
  pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
  H = pano_bottom - pano_top
  
  size = (W, H)
  
  
  
  # offset of first image relative to panorama
  X = int(min(top_left[0], bottom_left[0], 0))
  Y = int(min(top_left[1], top_right[1], 0))
  offset = (-X, -Y)

  return (size, offset)

genKeypointTask1()
genKeypointTask2()

best_match,kp1,kp2=generateKNN()
createPanoImage(best_match,kp1,kp2)

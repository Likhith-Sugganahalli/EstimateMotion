import numpy as np
import cv2
from matplotlib import pyplot as plt
import statistics as stats






def FeatureMatcher(image1,image2):
	i1 = cv2.imread(image1)          # queryImage
	i2 = cv2.imread(image2) # trainImage



	#cv.namedWindow('detected',cv.WINDOW_NORMAL)
	MIN_MATCH_COUNT = 10
	img1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)

	img2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

	height, width = img2.shape



	# Initiate SIFT detector
	sift = cv2.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.45*n.distance:
			good.append([m])

	print(len(good))

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					singlePointColor = None,
					flags = 2)

	# cv2.drawMatchesKnn expects list of lists as matches.
	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,**draw_params)
	plt.imshow(img3),plt.show()

	pts1 = []
	pts2 = []
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.8*n.distance:
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)

	findFundamentalMatrix(pts1,pts2,img1,img2)



	
def estimate_motion(match, kp1, kp2, k, depth1):
	"""
	Estimate camera motion from a pair of subsequent image frames

	Arguments:
	match -- list of matched features from the pair of images
	kp1 -- list of the keypoints in the first image
	kp2 -- list of the keypoints in the second image
	k -- camera calibration matrix 
	
	Optional arguments:
	depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

	Returns:
	rmat -- recovered 3x3 rotation numpy matrix
	tvec -- recovered 3x1 translation numpy vector
	image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
					 coordinates of the i-th match in the image coordinate system
	image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
					 coordinates of the i-th match in the image coordinate system
			   
	"""
	rmat = np.eye(3)
	tvec = np.zeros((3, 1))
	
	image1_points = []
	image2_points = []
	objectpoints = np.array([[],[],[]],dtype="float64")
	#i = 0
	print(len(match))
	for mi in match:       
		x1, y1 = kp1[mi.queryIdx].pt
		
		y1 = int(y1)
		x1 = int(x1)
		Z = depth1[int(y1), int(x1)]
		
		if Z < 1000:
			image1_points.append([x1, y1])
			
			x2, y2 = kp2[mi.trainIdx].pt
			y2 = int(y2)
			x2 = int(x2)
			image2_points.append([x2, y2])
			
			scaled_coord = np.dot(np.linalg.inv(k), np.array([x1, y1, 1]).reshape([3,1]))
			cali_coord = scaled_coord * Z
			#import pdb; pdb.set_trace()
			objectpoints = np.c_[objectpoints, cali_coord]
			#i = i + 1
		else:
			continue
			
			
	objectpoints = objectpoints.T
	imagepoints = np.array(image2_points,dtype="float32")
	dist_coef = np.zeros(4)
	#objectpoints = objectpoints.T
	#imagepoints = np.array(image2_points)
	print(objectpoints.shape)
	print(imagepoints.shape)
	print(len(image1_points))
	print(len(image2_points))
	ret,rmat,tvec = cv2.solvePnP(objectpoints,imagepoints, k,dist_coef)
	#import pdb; pdb.set_trace()
		
	### END CODE HERE ###
	
	return rmat, tvec, image1_points, image2_points



def findFundamentalMatrix(pts1,pts2,img1,img2):
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
	# We select only inlier points
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]

	print(F)
	# Find epilines corresponding to points in right image (second image) and
	# drawing its lines on left image
	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
	# Find epilines corresponding to points in left image (first image) and
	# drawing its lines on right image
	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
	plt.subplot(121),plt.imshow(img5)
	plt.subplot(122),plt.imshow(img3)

	plt.show()

def drawlines(img1,img2,lines,pts1,pts2):
	''' img1 - image on which we draw the epilines for the points in img2
		lines - corresponding epilines '''
	r,c = img1.shape
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
		img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
		img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
	return img1,img2




if __name__ == '__main__':
	image1 = ''
	image2 = ''
	FeatureMatcher()
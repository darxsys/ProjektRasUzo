import cv2 as cv
import numpy as np
import sys

############################################################################
# get_silhouette() parameters:
#
# 	personFile     - file containing an image with a person
# 	backgroundFile - file containing an image with the same background as
#					 in personFile
#
#	Optional:
#	 approach      - determines gray ('g') or colour ('c') approach
#				     	(default = 'c')
# 	 threshType    - determines whether median ('m') or mean ('a') is used 
#					 for image thresholding
#					 	(default = 'm')
# 	 threshold 	   - value added to median/mean for image thresholding 
#				     	(default = 25)
# 	 outFile       - silhouette output file
#			         	(default = silhouette is shown for 5 seconds)
#
############################################################################


############################################################################
# PUBLIC

def get_silhouette(personFile, backgroundFile, approach = 'c', threshType = 'm',
	threshold = 25, outFile = None):

	if approach == 'g':
		diff = _gray_approach(personFile, backgroundFile)
	else:
		diff = _color_approach(personFile, backgroundFile)

	_get_largest_contour(diff, threshType, threshold, outFile)

############################################################################


############################################################################
# PRIVATE

def _color_approach(personFile, backgroundFile):

	# reading images from input files

	person = cv.imread(personFile)
	_check(person, error = "no person file found")

	background = cv.imread(backgroundFile)
	_check(background, error = "no background file found")

	# spliting images to red, green & blue components

	personC = [person[:,:,0], person[:,:,1], person[:,:,2]]
	backgroundC = [background[:,:,0], background[:,:,1], background[:,:,2]]

	# subtracting images by component

	diffC = []

	diffC.append(cv.absdiff(personC[0], backgroundC[0]))
	diffC.append(cv.absdiff(personC[1], backgroundC[1]))
	diffC.append(cv.absdiff(personC[2], backgroundC[2]))

	# applying Gaussian blur to each component (reducing noise)

	diffC[0] = cv.GaussianBlur(diffC[0], (5, 5), 0)
	diffC[1] = cv.GaussianBlur(diffC[1], (5, 5), 0)
	diffC[2] = cv.GaussianBlur(diffC[2], (5, 5), 0)

	# merging components to a grey image 
	# cv.add() is a saturated operation (250 + 10 = 260 => 255)

	diff = cv.add(cv.add(diffC[0], diffC[1]), diffC[2])

	# applying Gaussian blur again

	diffB = cv.GaussianBlur(diff, (11, 11), 0)

	return diffB

def _gray_approach(personFile, backgroundFile):

	# reading images from input files in gray scale

	person = cv.imread(personFile, cv.CV_LOAD_IMAGE_GRAYSCALE)
	_check(person, error = "no person file found")

	background = cv.imread(backgroundFile, cv.CV_LOAD_IMAGE_GRAYSCALE)
	_check(background, error = "no background file found")

	# applying Gaussian blur to images

	personB = cv.GaussianBlur(person, (5, 5), 0)
	backgroundB = cv.GaussianBlur(background, (5, 5), 0)

	# subtracting images

	diff = cv.absdiff(personB, backgroundB)

	# applying Gaussian blur again

	diffB = cv.GaussianBlur(diff, (11, 11), 0)

	return diffB

def _get_largest_contour(diff, threshType, threshold, outFile):

	# calculating median/mean

	tr = threshold

	if threshType == 'a':
		tr += _get_mean(diff)
	else:
		tr += _get_median(diff)

	# thresholding the image (result is a black & white image)

	ret, diffT = cv.threshold(diff, tr, 255, cv.THRESH_BINARY)

	# finding the largest contour

	contours, hieararchy = cv.findContours(diffT, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

	max_ = [0, 0]
	rows, cols = diff.shape

	for i in range(len(contours)):
		temp = np.zeros((rows, cols, 1), np.uint8)
		cv.drawContours(temp, contours, i, (255, 255, 255), -1)
		size = cv.countNonZero(temp)

		if max_[1] < size:
			max_ = [i, size]

	silhouette = np.zeros((rows, cols, 1), np.uint8)
	cv.drawContours(silhouette, contours, max_[0], (255, 255, 255), -1)

	# saving image to file / showing it in a window

	if outFile is None:
		cv.namedWindow('silhouette', 0)
		cv.imshow('silhouette', silhouette)
		cv.waitKey(5000)
		cv.destroyAllWindows()
	else:
		cv.imwrite(outFile, silhouette)

def _get_median(image):
	# finds the median intensity value of an image

	arr = []
	rows, colls = image.shape
	for i in range(rows):
		for j in range(colls):
			arr.append(image.item(i, j))

	return np.median(arr)

def _get_mean(image):
	# finds the mean intensity value of an image

	arr = []
	rows, colls = image.shape
	for i in range(rows):
		for j in range(colls):
			arr.append(image.item(i, j))

	return np.mean(arr)

def _check(var, error):
	# checks if the variable is none

	if var is None:
		print(error)
		sys.exit(1)

############################################################################
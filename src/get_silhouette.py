import cv2 as cv
import numpy as np
import sys

############################################################################
# get_silhouette() parameters:
#   person          - image of a person and a background
#   background      - image of the same background
#
#   Optional:
#    approach       - determines gray ('g') or colour ('c') approach
#                       (default = 'c')
#    thresh_type    - determines whether median ('m') or mean ('a') is used 
#                     for image thresholding
#                       (default = 'm')
#    threshold      - value added to median/mean for image thresholding 
#                       (default = 25)
#    out_file       - determines whether the silhouette is saved to a file
#                       (default = None)
#
############################################################################


############################################################################
# PUBLIC

def get_silhouette(person, background, approach = 'c', thresh_type = 'm',
	threshold = 25, out_file = None):

	# subtracting the background to get the person

	if approach == 'g':
		diff = _gray_approach(person, background)
	else:
		diff = _colour_approach(person, background)

	# finding the silhouette

	return _get_largest_contour(diff, thresh_type, threshold, out_file)

############################################################################


############################################################################
# PRIVATE

def _colour_approach(person, background):

	# spliting images to red, green & blue components

	person_c = [person[:,:,0], person[:,:,1], person[:,:,2]]
	background_c = [background[:,:,0], background[:,:,1], background[:,:,2]]

	# subtracting images by component

	diff_c = []

	diff_c.append(cv.absdiff(person_c[0], background_c[0]))
	diff_c.append(cv.absdiff(person_c[1], background_c[1]))
	diff_c.append(cv.absdiff(person_c[2], background_c[2]))

	# applying Gaussian blur to each component (reducing noise)

	diff_c[0] = cv.GaussianBlur(diff_c[0], (5, 5), 0)
	diff_c[1] = cv.GaussianBlur(diff_c[1], (5, 5), 0)
	diff_c[2] = cv.GaussianBlur(diff_c[2], (5, 5), 0)

	# merging components to a grey image 
	# cv.add() is a saturated operation (250 + 10 = 260 => 255)

	diff = cv.add(cv.add(diff_c[0], diff_c[1]), diff_c[2])

	# applying Gaussian blur again

	diff_b = cv.GaussianBlur(diff, (11, 11), 0)

	return diff_b

def _gray_approach(person, background):

	# converting images to gray scale

	person_g = cv.cvtColor(person, cv.COLOR_BGR2GRAY)
	background_g = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

	# applying Gaussian blur to images

	person_b = cv.GaussianBlur(person_g, (5, 5), 0)
	background_b = cv.GaussianBlur(background_g, (5, 5), 0)

	# subtracting images

	diff = cv.absdiff(person_b, background_b)

	# applying Gaussian blur again

	diff_b = cv.GaussianBlur(diff, (11, 11), 0)

	return diff_b

def _get_largest_contour(diff, thresh_type, threshold, out_file):

	# calculating median/mean

	thr = threshold

	if thresh_type == 'a':
		thr += _get_mean(diff)
	else:
		thr += _get_median(diff)

	# thresholding the image (result is a black & white image)

	ret, diff_t = cv.threshold(diff, thr, 255, cv.THRESH_BINARY)

	# finding the largest contour

	contours, hieararchy = cv.findContours(diff_t, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

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

	# saving image to file if a file is given

	if out_file is not None:
		cv.imwrite(out_file, silhouette)

	# returning the silhouette

	return silhouette

# finds the median intensity value of a gray image
def _get_median(image):
	arr = []
	rows, colls = image.shape
	for i in range(rows):
		for j in range(colls):
			arr.append(image.item(i, j))

	return np.median(arr)

# finds the mean intensity value of a gray image
def _get_mean(image):
	arr = []
	rows, colls = image.shape
	for i in range(rows):
		for j in range(colls):
			arr.append(image.item(i, j))

	return np.mean(arr)

############################################################################
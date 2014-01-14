import numpy as np
import cv2 as cv
import math
import sys

# show_contour() parameters:
#   im          - image on which we want to draw contours
#   countours   - array of countours found on image im
#
#   Creates a new window with the image im that has drawn contours on itself
def show_contour(im, contours):
    # show the contour on the silhouette
    cv.drawContours(im, contours, -1, (0, 255, 0), 3)
    cv.imshow('Contour', im)
    # waiting for the SO to view the contour on the silhouette
    cv.waitKey(0)
    cv.destroyAllWindows()

# load_image_from_file() parameters:
#   filename    - path to the image we want to load
#   @return     - returns the loaded image
#
# Loads image from a file into opencv data structure
def load_image_from_file(filename):
    print "Loading %s" % (filename)
    return cv.imread(filename)

# get_features() parameters:
#   im      - image from we want to extract features
#   method  - determines what method to use, 0 for Granlunds coefficients, 1
#             for Hus moments
#   @return - 2D matrix containing features
#
# Extract features from an image
def get_features(im, method = 0):
    # opencv magic
    ret,thresh = cv.threshold(im, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        print "Couldn't find a contour :("
        sys.exit(1)

    # show_contour(im, contours)

    # if we find more than 1 contour, continue computation with the largest one
    silhouette = None

    for cnt in contours:
        if silhouette is None or cv.contourArea(silhouette) < cv.contourArea(cnt):
            silhouette = cnt

    if method == 0:
        return _get_granlund_coefficients(silhouette)
    else:
        return _get_hu_coefficients(silhouette)

def _get_hu_coefficients(silhouette):
    # some more opencv magic
    mom = cv.moments(silhouette)
    hu = cv.HuMoments(mom)

    # convert to appropriate record
    coefficients = list()
    for xs in hu.tolist():
        for x in xs:
            coefficients.append(x)

    return np.array([coefficients], dtype=np.float32)

def _get_granlund_coefficients(silhouette):
    # please make sure this interval contains 1
    FOURIER_MIN_ID = -3
    FOURIER_MAX_ID = 4

    # prepare values to be consistent with "meth"
    T = len(silhouette)
    fourier_indices = range(FOURIER_MIN_ID, FOURIER_MAX_ID + 1)
    fourier_coefficients = dict()

    for n in fourier_indices:
        Re = 0.0
        Im = 0.0

        for (d, t) in zip(silhouette, range(len(silhouette))):
            x = d.item(0)
            y = d.item(1)

            A = 2.0 * math.pi * n * t / T

            Re += x * math.cos(A) + y * math.sin(A)
            Im += y * math.cos(A) - x * math.sin(A)

        Re /= T
        Im /= T

        fourier_coefficients[n] = complex(Re, Im)

    # print fourier_coefficients

    # compute Granlund coefficients
    granlund_coefficients = list()

    for p in range(1, FOURIER_MAX_ID):
        for q in range(2, 2 - FOURIER_MIN_ID):
    # for (p,q) in [(1,1), (2,2), (2,3), (2,4), (2,5), (3,3), (4,3), (4,4)]:
            d_pq  = fourier_coefficients[p + 1] ** q
            d_pq *= fourier_coefficients[1 - q] ** p
            d_pq /= fourier_coefficients[1]     ** (p + q)
            granlund_coefficients.append(d_pq)

    # granlund coefficients are a list of complex numbers
    granlund_coefficients_list = list()
    for c in granlund_coefficients:
        granlund_coefficients_list.append(c.real)
        granlund_coefficients_list.append(c.imag)

    # print granlund_coefficients
    return np.array([granlund_coefficients_list], dtype=np.float32)


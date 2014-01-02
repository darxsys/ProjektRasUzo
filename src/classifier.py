import sys
import cv2 as cv

__doc__ = """Wrapper to create new classifiers from OpenCV or other libraries.
"""

def normal_bayes(dataset, responses):
    """Returns a trained OpenCV Normal Bayes Classifier.
    More info: http://docs.opencv.org/modules/ml/doc/normal_bayes_classifier.html
    """

    classifier = cv.NormalBayesClassifier()
    classifier.train(dataset, responses)
    return classifier

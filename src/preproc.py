import sys
import os
import numpy as np
import cv2 as cv

import get_silhouette
import granlund

__doc__ == """Module with helper functions for
dataset and other preprocessing for classifiers.
"""

def display_image(im):
    cv.namedWindow("Disp", 0)
    cv.imshow("Disp", im)
    cv.waitKey(0)
    return

def prepare_dataset_cv(folder, threshold_, method_):
    """For a given folder, goes through all the subfolders that contain
    images for a particular person and transforms them into features.
    Also creates a dictionary later used for decoding of a persons name.
    Returns dataset, labels, dictionary - in that order.
    """

    dataset = None
    # dataset = np.array([[]], dtype=np.float32)
    # labels = np.array([], dtype=np.float32)
    labels = []
    labels_dict = {}
    counter = -1

    # folder_path = os.path.join(folder, '/')
    subfolders = os.listdir(folder)
    for sfolder in subfolders:
        if not os.path.isdir(os.path.join(folder, sfolder)): continue

        counter += 1
        labels_dict[counter] = sfolder

        pics_path = os.path.join(folder, sfolder, 'pics/')
        backs_path = os.path.join(folder, sfolder, 'back/')

        images = os.listdir(pics_path)
        for image in images:
            if not (image.endswith(".jpg") or image.endswith(".JPG")): continue

            labels.append(counter)

            im = granlund.load_image_from_file(os.path.join(pics_path, image))
            background = granlund.load_image_from_file(os.path.join(backs_path, image))
            silh = get_silhouette.get_silhouette(im, background, threshold = threshold_ , approach='c')
            # display_image(silh)

            features = granlund.get_features(silh, method=method_)

            if dataset is None:
                dataset = features
            else:
                dataset = np.vstack((dataset, features))

    labels = np.array(labels, dtype = np.float32)
    return dataset, labels, labels_dict

def preproc_dataset_cv(path):
    """Reads a dataset from the path and returns two numpy arrays:
    train_data and responses for a given dataset. Intended for OpenCV
    classifiers.
    """

    with open(path, "r") as f:
        data = []
        responses = []

        for line in f:
            if line == "\n":
                continue
            line = line.strip().split()
            nums = [float(x) for x in line[:-1]]
            data.append(tuple(nums))
            responses.append(float(line[-1]))

    data = np.matrix(data, np.float32)
    responses = np.matrix(responses, np.float32)
    return data, responses

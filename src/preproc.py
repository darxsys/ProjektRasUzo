import sys
import os
import numpy as np

import get_silhouette
import granlund

__doc__ == """Module with helper functions for
dataset and other preprocessing for classifiers.
"""

def prepare_dataset_cv(folder):
    """For a given folder, goes through all the subfolders that contain
    images for a particular person and transforms them into characteristics.
    Also creates a dictionary later used for decoding of a persons name.
    Returns dataset, labels, dictionary - in that order.
    """

    dataset = np.array([[]], dtype=np.float32)
    labels = np.array([], dtype=np.float32)
    labels_dict = {}
    counter = -1

    subfolders = os.listdir(folder)
    for sfolder in subfolders:
        counter += 1
        labels_dict[counter] = sfolder

        images = os.listdir(os.path.join(folder, sfolder))
        for image in images:
            silh = get_silhouette.get_silhouette(image)








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

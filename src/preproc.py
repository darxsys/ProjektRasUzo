import sys
import numpy as np

__doc__ == """Module with helper functions for
dataset and other preprocessing for classifiers.
"""

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
            responses.append(list((float(line[-1]),)))

    data = np.matrix(data, np.float32)
    responses = np.matrix(responses, np.float32)
    return data, responses

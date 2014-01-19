import sys
import cv2 as cv

__doc__ = """Wrapper to create new classifiers from OpenCV or other libraries.
"""

class NormalBayes(object):
    """Wraps a trained OpenCV Normal Bayes Classifier.
    More info: http://docs.opencv.org/modules/ml/doc/normal_bayes_classifier.html
    """    

    def __init__(self):
        self.model = cv.NormalBayesClassifier()

    def train(self, dataset, responses):
        """Dataset and responses are assumed to be a 2D and 1D numpy matrix of type np.float32.
        """

        self.model.train(dataset, responses)

    def predict(self, samples):
        """Samples have to be a 2D numpy array of type np.float32.
        Returns a list of prediction values.
        """

        pred_results = self.model.predict(samples)
        return [int(x[0]) for x in pred_results[1]]

class KNN(object):
    """Wraps a trained OpenCV k_nn classifier.
    More info: http://docs.opencv.org/modules/ml/doc/k_nearest_neighbors.html
    """

    def __init__(self):
        self.model = cv.KNearest()
        self.max_K = 32

    def train(self, dataset, responses, params):
        """Dataset and responses are assumed to be a 2D and 1D numpy matrix of type np.float32.
        Additionally, optional max_neighbors argument can be provided.
        """

        if "nmax" in params:
            self.max_K = params["nmax"]
        else:
            self.max_K = 32
            
        self.model.train(dataset, responses, maxK=self.max_K)

    def predict(self, samples, params):
        """Accepts samples for classification and K - number of neighbors to use. 
        Notice: K has to be <= maxK that was set while training. 
        Refer here: http://docs.opencv.org/modules/ml/doc/k_nearest_neighbors.html
        for more info. Samples are 2D numpy array of type np.float32.
        Returns a list of prediction values.
        """

        if "nclass" in params:
            K = params["nclass"]
        else:
            K = 7

        if K > self.max_K:
            print ("Bad argument: K")
            return []

        out = self.model.find_nearest(samples, K)
        return [int(x[0]) for x in out[1]]

class RandomTrees(object):
    """Wraps a trained OpenCV RTrees classifier.
    More info: http://docs.opencv.org/modules/ml/doc/random_trees.html
    """

    def __init__(self):
        self.model = cv.RTrees()

    def train(self, dataset, responses, params):
        """Dataset and responses are assumed to be a 2D and 1D numpy matrix of type np.float32.
        max_d corresponds to the max tree depth. Parameter criteria can be:
        --CV_TERMCRIT_ITER Terminate learning by the max_num_of_trees_in_the_forest;
        --CV_TERMCRIT_EPS Terminate learning by the forest_accuracy;
        --CV_TERMCRIT_ITER + CV_TERMCRIT_EPS Use both termination criteria.
        Refer here: http://docs.opencv.org/modules/ml/doc/random_trees.html
        """

        if "maxdepth" in params:
            max_d = params["maxdepth"]
        else:
            max_d = 4
        if "criteria" in params:
            criteria = params["criteria"]
        else:
            criteria=cv.TERM_CRITERIA_MAX_ITER+cv.TERM_CRITERIA_EPS
        if "maxerror" in params:
            max_error = params["maxerror"]
        else:
            max_error = 0.1
        if "maxtrees" in params:
            max_num_trees = params["maxtrees"]
        else:
            max_num_trees = 10

        parameters = dict(max_depth=max_d, min_sample_count=1, use_surrogates=False, 
            calc_var_importance=True, max_categories=10, nactive_vars=0, 
            term_crit=(criteria, max_num_trees, max_error)) # not sure if max_error belongs here :D
        self.model.train(dataset, cv.CV_ROW_SAMPLE, responses, params=parameters)
        # print ("Num of trees: " + str(self.model.getVarImportance()))

    def predict(self, samples):
        """Returns a list of prediction values for all samples.
        Assuming samples are 2D numpy array of type np.float32.
        """

        return [int(self.model.predict(s)) for s in samples]

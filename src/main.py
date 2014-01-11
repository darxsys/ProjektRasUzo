import sys
import numpy as np

import preproc
import classifier
import granlund
import get_silhouette

def train(path):
    """Given a path to the pictures, trains all possible classifiers.
    """

    bayes = classifier.NormalBayes()
    knn = classifier.KNN()
    tree = classifier.RandomTrees()

    dataset, responses, decode = preproc.prepare_dataset_cv(path)
    bayes.train(dataset, responses)
    knn.train(dataset, responses)
    tree.train(dataset, responses)

    return bayes, knn, tree, decode

def predict(bayes, knn, tree, decode):
    """Accepts trained classifiers and decode dictionary. Waits for a path to a picture to classify.
    """

    print ("Type in the path to a picture and its background. -1 to end.")
    while True:
        pic = raw_input()
        if pic == "-1":
            break

        back = raw_input()
        image = granlund.load_image_from_file(pic)
        background = granlund.load_image_from_file(back)
        silh = get_silhouette.get_silhouette(image, background)
        features = granlund.get_features(silh)

        print ("Bayes result: " + decode[bayes.predict(features)[0]])
        print ("KNN result: " + decode[knn.predict(features)[0]])
        print ("Tree result: " + decode[tree.predict(features)[0]])

if __name__ == "__main__":
    if not len(sys.argv) == 2:
        raise ValueError("Only one argument - path to pictures.")

    bayes, knn, tree, decode = train(sys.argv[1])
    predict(bayes, knn, tree, decode)

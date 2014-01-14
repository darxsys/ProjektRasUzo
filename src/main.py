import sys
import numpy as np
import cv2 as cv

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

    print decode
    N = 0
    bayes_correct = 0
    knn_correct = 0
    tree_correct = 0

    print ("Type in the path to a picture and its background. -1 to end.")
    print

    while True:
        line = sys.stdin.readline()
        pic = line.split()[0]
        if pic == "-1":
            break
        class_ = line.strip().split()[1]

        line = sys.stdin.readline()
        back = line.strip()

        image = granlund.load_image_from_file(pic)
        background = granlund.load_image_from_file(back)
        silh = get_silhouette.get_silhouette(image, background, threshold = 150)
        # preproc.display_image(silh)
        features = granlund.get_features(silh, method=1)

        bayes_res = decode[bayes.predict(features)[0]]
        knn_res = decode[knn.predict(features, 7)[0]]
        tree_res = decode[tree.predict(features)[0]]

        N += 1
        if bayes_res == class_:
            bayes_correct += 1
        if knn_res == class_:
            knn_correct += 1
        if tree_res == class_:
            tree_correct += 1

        print ("Bayes result: " + bayes_res)
        print ("KNN result: " + knn_res)
        print ("Tree result: " + tree_res)

    print

    print("Correctness:")
    print("Bayes: %.5lf" % (bayes_correct / float(N)))
    print("KNN: %.5lf" % (knn_correct / float(N)))
    print("tree: %.5lf" % (tree_correct / float(N)))

if __name__ == "__main__":
    if not len(sys.argv) == 2:
        raise ValueError("Only one argument - path to pictures.")

    bayes, knn, tree, decode = train(sys.argv[1])
    predict(bayes, knn, tree, decode)

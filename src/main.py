import sys
import numpy as np
import cv2 as cv
import getopt

import preproc
import classifier
import granlund
import get_silhouette

def get_arguments(argv):
    # print(argv)
    try:
        opts, args = getopt.getopt(argv,"t:m:p:s",["path=","threshold=","method=","parameters="])
    except getopt.GetoptError:
        print ('main.py --path=<path_to_data> --method=<hu>|<granlund>  --threshold=<floating_point_number> --parameters=<0>|<1>')

    # training = ""
    path = None
    threshold = None
    method = None
    parameters = None
    # generalization = ""

    for opt, arg in opts:
        # print (opt)
        if opt == '-h':
            print ('main.py --method=<hu>|<granlund>  --threshold=<floating_point_number>')
            sys.exit(1)

        elif opt == "--threshold":
            threshold = float(arg)
        elif opt == "--method":
            if arg == "hu":
                method = 1
            else:
                method = 0
            # method = arg
        elif opt == "--path":
            path = arg
        elif opt == "--parameters":
            parameters = int(arg)

    return path, threshold, method, parameters

def train(path, threshold, method, parameters):
    """Given a path to the pictures, trains all possible classifiers.
    """

    bayes = classifier.NormalBayes()
    knn = classifier.KNN()
    tree = classifier.RandomTrees()

    dataset, responses, decode = preproc.prepare_dataset_cv(path, threshold, method, parameters)
    bayes.train(dataset, responses)
    knn.train(dataset, responses)
    tree.train(dataset, responses)

    return bayes, knn, tree, decode

def predict(bayes, knn, tree, decode, threshold_, method_, parameters):
    """Accepts trained classifiers and decode dictionary. Waits for a path to a picture to classify.
    """

    print decode
    N = 0
    bayes_correct = 0
    knn_correct = 0
    tree_correct = 0

    print ("Type in the path to a picture and its background. -1 to end.")
    print ("Optionally, you can also put parameters if you selected it.")
    print

    while True:
        line = sys.stdin.readline()
        pic = line.split()[0]
        if pic == "-1":
            break
        class_ = line.strip().split()[1]

        line = sys.stdin.readline()
        back = line.strip()

        if parameters == 1:
            threshold_ = float(sys.stdin.readline().strip())

        image = granlund.load_image_from_file(pic)
        background = granlund.load_image_from_file(back)
        silh = get_silhouette.get_silhouette(image, background, threshold = threshold_)
        # preproc.display_image(silh)
        features = granlund.get_features(silh, method=method_)

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
    path, threshold, method, parameters = get_arguments(sys.argv[1:])
    if path == None or threshold == None or method == None:
        raise ValueError("Not enough arguments.")

    bayes, knn, tree, decode, = train(path, threshold, method, parameters)
    predict(bayes, knn, tree, decode, threshold, method, parameters)

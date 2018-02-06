from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import argparse
import time
import cv2

def load_digits(datasetPath):
	X = np.genfromtxt(datasetPath, delimiter=",", dtype="uint8")
	y = X[:, 0]
	X = X[:, 1:]
	return (X, y)

def scale(X, eps=0.001):
	return (X - np.min(X, axis=0)) / (np.max(X, axis=0) + eps)

def nudge(X, y):
	translations = [(0, -1), (0, 1), (-1, 0), (1, 0)]
	data = []
	target = []
	
	for (image, label) in zip(X, y):
		image = image.reshape(28, 28)
		for (tX, tY) in translations:
			M = np.float32([[1, 0, tX], [0, 1, tY]])
			trans = cv2.warpAffine(image, M, (28, 28))
			data.append(trans.flatten())
			target.append(label)
	
	return (np.array(data), np.array(target))

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True, help="path to the dateset file")
	ap.add_argument("-t", "--test", required=True, type=float, help="size of test split")
	ap.add_argument("-s", "--search", type=int, default=0, help="whether or not a grid search should be performed")
	args = vars(ap.parse_args())
	
	(X, y) = load_digits(args["dataset"])
	X = X.astype("float32")
	X = scale(X)
	
	(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=args["test"], random_state=42)
	
	if args["search"]==1:
		print "SEARCHING LOGISTIC REGRESSION"
		params = {"C": [1.0, 10.0, 100.0]}
		start = time.time()
		gs = GridSearchCV(LogisticRegression(), params, n_jobs=-1, verbose=1)
		gs.fit(trainX, trainY)
		
		print("done in %0.3fs" % (time.time() - start))
		print "best score: %0.3f" % (gs.best_score_)
		print "LOGISTIC REGRESSION PARAMETERS"
		bestParams = gs.best_estimator_.get_params()
		
		for p in sorted(params.keys()):
			print "\t %s: %f" % (p, bestParams[p])
		
		rbm = BernoulliRBM()
		logistic = LogisticRegression()
		classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
		
		print "SEARCHING RBM + LOGISTIC REGRESSION"
		params = {
			"rbm__learning_rate": [0.1, 0.01, 0.001],
			"rbm__n_iter": [20, 40, 80],
			"rbm__n_components": [50, 100, 200],
			"logistic__C": [1.0, 10.0, 100.0]}
		start = time.time()
		gs = GridSearchCV(classifier, params, n_jobs=-1, verbose=1)
		gs.fit(trainX, trainY)
		
		print "\ndone in %0.3fs" % (time.time() - start)
		print "best score: %0.3f" % (gs.best_score_)
		print "RBM + LOGISTIC REGRESSION PARAMETERS"
		bestParams = gs.best_estimator_.get_params()
		
		for p in sorted(params.keys()):
			print "\t %s: %f" % (p, bestParams[p])
		
		print "\nIMPORTANT"
		print "Now that your parameters have been searched, manually set"
		print "them and re-run this script with --serach 0"
	else:
		logistic = LogisticRegression(C=1.0)
		logistic.fit(trainX, trainY)
		print "LOGISTIC REGRESSION ON ORIGINAL DATASET"
		print classification_report(testY, logistic.predict(testX))
		
		rbm = BernoulliRBM(n_components=200, n_iter=20, learning_rate=0.01, verbose=True)
		logistic = LogisticRegression(C=1.0)
		
		classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
		classifier.fit(trainX, trainY)
		print "RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET"
		print classification_report(testY, classifier.predict(testX))
		
		print "RBM + LOGISTIC REGRESSION ON NUDGED DATASET"
		(testX, testY) = nudge(testX, testY)
		print classification_report(testY, classifier.predict(testX))
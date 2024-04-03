# minicontest.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
from sklearn.neighbors import KNeighborsClassifier
"""
MAKE SURE YOU HAVE SCIKIT-LEARN INSTALLED!!!! 
aka; pip install scikit-learn b4 you run my code
"""

class contestClassifier(classificationMethod.ClassificationMethod):
    """
    using a k-NN classifier thru the scikit-learn library
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "minicontest"
        self.classifier = KNeighborsClassifier(n_neighbors=3)

    def train(self, data, labels, validationData, validationLabels):
        """
        Train the k-NN classifier using the provided training data and labels.
        """
        # Convert the training data and labels to the format required by scikit-learn
        X_train = [datum.values() for datum in data]
        y_train = labels

        # Train the k-NN classifier
        self.classifier.fit(X_train, y_train)

        # Evaluate the classifier on the validation data
        guesses = self.classify(validationData)
        correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
        accuracy = 100.0 * correct / len(validationLabels)
        print "Validation set accuracy: %.2f%%" % accuracy

    def classify(self, testData):
        """
        Classify the test data using the trained k-NN classifier.
        """
        # Convert the test data to the format required by scikit-learn
        X_test = [datum.values() for datum in testData]

        # Make predictions using the trained classifier
        guesses = self.classifier.predict(X_test)

        return guesses

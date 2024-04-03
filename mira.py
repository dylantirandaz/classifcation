# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util

PRINT = True

class MiraClassifier:
    """
    Mira classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."
        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA. Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.
        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        bestWeights = None
        bestAccuracy = 0
        
        for C in Cgrid:
            self.initializeWeightsToZero()
            
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):
                    datum = trainingData[i]
                    label = trainingLabels[i]

                    # Compute the scores for each label using the current weights
                    scores = util.Counter()
                    for l in self.legalLabels:
                        scores[l] = self.weights[l] * datum

                    # Find the predicted label with the highest score
                    predicted_label = scores.argMax()

                    # If the predicted label is incorrect, update the weights using MIRA
                    if predicted_label != label:
                        # Compute the difference between the scores of the predicted and true labels
                        diff = scores[predicted_label] - scores[label]
                        
                        # Compute the update magnitude using the MIRA formula
                        tau = min(C, diff / (2 * (datum * datum)))
                        
                        # Update the weights
                        self.weights[label] += tau * datum
                        self.weights[predicted_label] -= tau * datum

            # Evaluate the current weights on the validation data
            guesses = self.classify(validationData)
            accuracy = sum(guesses[i] == validationLabels[i] for i in range(len(validationLabels))) / float(len(validationLabels))
            
            # If the current weights have the best accuracy so far, store them
            if accuracy > bestAccuracy:
                bestWeights = self.weights.copy()
                bestAccuracy = accuracy

        # Set the final weights to the best weights found during tuning
        self.weights = bestWeights

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. See the project description for details.
        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns a list of the 100 features with the greatest difference in feature values
        w_label1 - w_label2
        """
        featuresOdds = [(feature, self.weights[label1][feature] - self.weights[label2][feature])
                        for feature in self.features]
        featuresOdds.sort(key=lambda x: -abs(x[1]))
        return [feature for feature, odds in featuresOdds[:100]]

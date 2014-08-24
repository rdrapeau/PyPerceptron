"""
This is an implentation of a Perceptron (http://en.wikipedia.org/wiki/Perceptron).

@author Ryan Drapeau
"""
from numpy import alterdot, array, dot

class Perceptron(object):

    def __init__(self, number_of_classes, dimension):
        self.weights = []
        for i in xrange(number_of_classes):
            self.weights.append([0] * (dimension + 1))

        self.learning_rate = 0.1
        self.max_iterations = 100
        self.dimension = dimension


    def set_learning_rate(self, learning_rate):
        """
        Sets the learning rate of the update step.

        learning_rate (float) - The learning rate of the update step
        """
        self.learning_rate = learning_rate


    def set_max_iterations(self, max_iterations):
        """
        Sets the max iterations the algorithm will take when training.

        max_iterations (int) - The number of iterations to stop at
        """
        self.max_iterations = max_iterations


    def set_weights(self, weights):
        """
        Sets the weight matrix the algorithm will use.

        weights (Array[(Array[(float)])]) - The weight matrix to use
        """
        self.dimension = len(weights[0]) - 1
        self.weights = weights


    def train_set(self, training_samples, training_labels):
        """
        Trains the Perceptron on the given sample vectors and labels.

        training_samples (Array[(float)]) - The sample vectors to train on
        training_labels (Array[(int)]) - The labels of the sample vectors
        @throws (Exception) if the training samples and labels do not have the same dimension
        """
        print 'Training Data'
        if len(training_samples) != len(training_labels) or len(training_samples) < 1:
            raise Exception("Samples and labels must have the same number of entries")

        finished = False
        count = 0
        training_samples = [array([1] + sample) for sample in training_samples]
        while not finished and count < self.max_iterations:
            finished = True
            count += 1
            for i, sample in enumerate(training_samples):
                finished &= self.train(sample, training_labels[i])

            print 'Finished Pass', count

        print 'Finished Training'


    def train(self, sample, label):
        """
        Trains a single sample.

        sample (Array[float]) - The sample vector
        label (int) - The label class for this sample
        """
        max_class = array([dot(sample, weight) for weight in self.weights]).argmax()

        if max_class != label:
            self.__update_weights(label, max_class, sample)
            return False

        return True


    def __update_weights(self, label, max_class, sample):
        """
        Updates the weight vector with the error and the sample.

        label (int) - The true label class for this sample
        max_class (int) - The incorrect label class for this sample
        sample (Array[(float)]) - The sample vector
        """
        for i in xrange(len(self.weights[max_class])):
            self.weights[max_class][i] -= self.learning_rate * sample[i]

        for i in xrange(len(self.weights[label])):
            self.weights[label][i] += self.learning_rate * sample[i]


    def classify(self, sample):
        """
        Classifies the sample.

        sample (Array[(float)]) - The vector to classify
        @throws (Exception) if the Perceptron was not trained or if the vector's dimension is incorrect
        @return (int) The classification of the input vector
        """
        if self.weights == None or len(sample) != self.dimension:
            raise Exception("Illegal dimension or not trained")

        sample = array([1] + sample)
        return array([dot(sample, weight) for weight in self.weights]).argmax()

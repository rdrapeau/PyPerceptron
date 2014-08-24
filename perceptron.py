"""
This is an implentation of a Perceptron (http://en.wikipedia.org/wiki/Perceptron).

@author Ryan Drapeau
"""
from numpy import dot, array

class Perceptron(object):

    def __init__(self):
        self.weights = None
        self.learning_rate = 0.1
        self.max_iterations = 100
        self.dimension = None
        self.threshold = 0.5


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
        self.dimension = len(weights) - 1
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

        self.dimension = len(training_samples[0])
        self.weights = array([0] * (self.dimension + 1))
        finished = False
        count = 0
        training_samples = [array([1] + sample) for sample in training_samples]
        while not finished and count < self.max_iterations:
            finished = True
            count += 1
            for i, sample in enumerate(training_samples):
                finished &= self.train(sample, training_labels[i])

        print 'Finished Training'
        # print self.weights


    def train(self, sample, label):
        value = self.__step_function(dot(sample, self.weights))

        if self.__step_function(value) != self.__step_function(label):
            self.__update_weights(label - value, sample)
            return False

        return True


    def __step_function(self, value):
        """
        Evalutes the value to a step function based on the threshold.

        @return (int): class 1 if the value is above the threshold, otherwise 0
        """
        return 1 if value > self.threshold else 0


    def __update_weights(self, error, sample):
        """
        Updates the weight vector with the error and the sample.

        error (float) - The error to update with
        sample (Array[(float)]) - The sample vector
        """
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight + self.learning_rate * error * sample[i]


    def classify(self, input_vector):
        """
        Classifies the input_vector.

        input_vector (Array[(float)]) - The vector to classify
        @throws (Exception) if the Perceptron was not trained or if the vector's dimension is incorrect
        @return (int) The classification of the input vector
        """
        if self.weights == None or len(input_vector) != self.dimension:
            raise Exception("Illegal dimension or not trained")

        input_vector = [1] + input_vector
        value = dot(input_vector, self.weights)
        return self.__step_function(value)

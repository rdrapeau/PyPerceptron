import cProfile
from perceptron import *

PATH_TO_DATA_FOLDER = '../Datasets/MNIST/'
TEST_FILE_NAME = 'mnist_test.csv'
TRAIN_FILE_NAME = 'mnist_train.csv'


def convert_data(file_path):
    result = []
    with open(file_path, 'r') as f:
        lines = f.read().split('\n')[:-1]
        print 'Reading Data From File (', len(lines), 'lines )'
        for line in lines:
            parsed_line = [int(number) for number in line.split(',')]
            label = parsed_line[0]
            data = parsed_line[1:]
            line_result = (label, data)
            result.append(line_result)

    print 'Finished reading'
    return result


def convert_labels_to_binary(labels, target_class):
    print 'Converting To Binary'
    for i, label in enumerate(labels):
        if label == target_class:
            labels[i] = 1
        else:
            labels[i] = 0

    print 'Finished Converting To Binary'


def get_data(data):
    training_samples = []
    training_labels = []
    for entry in data:
        sample = entry[1]
        label = entry[0]
        training_samples.append(sample)
        training_labels.append(label)

    return (training_labels, training_samples)


def test_perceptron(test_labels, test_samples):
    print 'Testing Perceptron'
    count = 0
    for label, sample in zip(test_labels, test_samples):
        result = perceptron.classify(sample)
        if result == label:
            count += 1

    print 'Finished Testing Perceptron'
    return float(count) / len(test_labels)


perceptron = Perceptron()
perceptron.set_learning_rate(0.25)
perceptron.set_max_iterations(10)

training_data = get_data(convert_data(PATH_TO_DATA_FOLDER + TRAIN_FILE_NAME))
training_labels = training_data[0]
convert_labels_to_binary(training_labels, 1)
training_samples = training_data[1]

perceptron.train_set(training_samples, training_labels)

test_data = get_data(convert_data(PATH_TO_DATA_FOLDER + TEST_FILE_NAME))
test_labels = test_data[0]
convert_labels_to_binary(test_labels, 1)
test_samples = test_data[1]

print 'Accuracy:', test_perceptron(test_labels, test_samples)
print perceptron.weights
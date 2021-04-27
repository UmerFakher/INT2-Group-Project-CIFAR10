import numpy as np
import copy
import time
import math

################################################ images and labels

training_set1 = []
training_set2 = []
training_set3 = []
training_set4 = []
training_set5 = []
training_labels1 = []
training_labels2 = []
training_labels3 = []
training_labels4 = []
training_labels5 = []
testing_set = []
testing_labels = []

################################################ weights and biases

weights = [0] * 6
biases = [0] * 6


current_depth = 6


def initialise_w_and_b():

    global weights
    global biases

    weights[0] = np.random.rand(100, 3072)
    weights[1] = np.random.rand(100, 100)
    weights[2] = np.random.rand(100, 100)
    weights[3] = np.random.rand(100, 100)
    weights[4] = np.random.rand(100, 100)
    weights[5] = np.random.rand(10, 100)
    biases[0] = np.zeros(100)
    biases[1] = np.zeros(100)
    biases[2] = np.zeros(100)
    biases[3] = np.zeros(100)
    biases[4] = np.zeros(100)
    biases[5] = np.zeros(10)


layer_outputs = [0] * 6
learning_rate = 0.1
expected_output = [0] * 10


def main():
    global training_set1

    populate_images()
    initialise_w_and_b()

    start = time.time()
    train(training_set1, training_labels1)
    end = time.time()
    print(end - start, "seconds")


def train(images = [], labels = []):

    for i in range(0, 100):
        output = copy.deepcopy(images[i]) / 255

        for j in range(0, current_depth):
            output = forward_prop(output, j)
            normalization_factor = np.linalg.norm(output)
            if normalization_factor != 0:
                output = output / normalization_factor

            print(output)

        actual_results(labels[i])

        output_error = expected_output - output

        cost = cost_function(output)

        print("#####################################################################################")

        for k in range(current_depth - 2, -1, -1):
             output_error = backward_prop(copy.deepcopy(images[i]) / 255, output_error, k)

        print("#####################################################################################")


def forward_prop(x = [], layer = 0):           # pass the input through the layer of the neural network
    output = relu(np.dot(x, np.transpose(weights[layer])) + biases[layer])
    layer_outputs[layer] = output
    return output


def backward_prop(image = [], output_error = [], layer = 0):
    # pass the output and cost back through the neural network to change the weights and biases using gradient descent

    error = np.dot(((np.transpose(weights[layer + 1])) * output_error), relu_derivative(layer_outputs[layer + 1]))

    weights_gradient = layer_outputs[layer - 1] * error ######################
    biases_gradient = error

    weights[layer] = weights[layer] - np.transpose([(learning_rate * weights_gradient)])
    biases[layer] = biases[layer] - (learning_rate * biases_gradient)

    return error


def cost_function(results = []):
    
    cost = 0
    
    for i in range(0, len(results)):
        cost += ((expected_output[i] - results[i])**2) / 1

    return cost


def relu(x = []):
    output = ((np.absolute(x) + x) / 2)
    return output.astype(int)


def relu_derivative(x = [],):
    output = copy.deepcopy(x)
    for i in range(0, len(output)):
        if output[i] > 0:
            output[i] = 1
        else:
            output[i] = 0
    print(output)
    return output


def actual_results(label):
    
    for i in range(0, 10):
        if i == label:
            expected_output[i] = 1
        else:
            expected_output[i] = 0


def populate_images():
    global training_set1
    global training_set2
    global training_set3
    global training_set4
    global training_set5
    global training_labels1
    global training_labels2
    global training_labels3
    global training_labels4
    global training_labels5
    global testing_set
    global testing_labels

    data = unpickle("data_batch_1")
    training_set1 = data[b'data']
    training_labels1 = data[b'labels']
    data = unpickle("data_batch_2")
    training_set2 = data[b'data']
    training_labels2 = data[b'labels']
    data = unpickle("data_batch_3")
    training_set3 = data[b'data']
    training_labels3 = data[b'labels']
    data = unpickle("data_batch_4")
    training_set4 = data[b'data']
    training_labels4 = data[b'labels']
    data = unpickle("data_batch_5")
    training_set5 = data[b'data']
    training_labels5 = data[b'labels']
    data = unpickle("test_batch")
    testing_set = data[b'data']
    testing_labels = data[b'labels']


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        output = pickle.load(fo, encoding='bytes')
    return output


if __name__ == "__main__":
    main()
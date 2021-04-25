import numpy as np
import copy
import time

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

def initialise_w_and_b():

    global weights
    global biases

    weights[0] = np.random.randn(3072, 3072)
    weights[1] = np.random.randn(3072, 3072)
    weights[2] = np.random.randn(3072, 3072)
    weights[3] = np.random.randn(3072, 3072)
    weights[4] = np.random.randn(3072, 3072)
    weights[5] = np.random.randn(3072, 10)
    biases[0] = np.zeros(3072)
    biases[1] = np.zeros(3072)
    biases[2] = np.zeros(3072)
    biases[3] = np.zeros(3072)
    biases[4] = np.zeros(3072)
    biases[5] = np.zeros(10)


layer_outputs = [0] * 6



def main():
    global training_set1

    populate_images()
    initialise_w_and_b()

    output = training_set1[0]

    for i in range(0, 6):
        output = forward_prop(output, i)
        normalize_multiplier = max(output)
        output = output / normalize_multiplier

    cost = cost_function(output, training_labels1[0])

    for j in range(5, -1, -1):
        backward_prop(j, cost)

    print("yeet")


def forward_prop(x, layer):           # pass the input through the layer of the neural network
    output = relu(np.dot(x, weights[layer]) + biases[layer])
    return output


def backward_prop(layer, cost):
    # pass the output and cost back through the neural network to change the weights and biases
    output = np.dot(layer_outputs[layer], cost * relu_derivative(layer_outputs[layer], layer))


def cost_function(results, actual_value):
    
    cost = 0
    
    for i in range(0, len(results)):
        if i == actual_value:
            cost += (1 - results[i])
        else:
            cost += (0 - results[i])

    cost = cost / len(results)

    return cost



def relu(x):
    output = ((np.absolute(x) + x) / 2)
    return output.astype(int)


def relu_derivative(x, layer):
    
    output = (x + np.absolute(x)) / (2 * x)
    output = np.dot(output, weights[layer])
    return output



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


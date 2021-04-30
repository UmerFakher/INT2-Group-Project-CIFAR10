# INT2-Group-Project-CIFAR10

- [The CIFAR-10 Dataset Link](https://www.cs.toronto.edu/~kriz/cifar.html)


## **NN_Naive_ALLBatch_Refactored.ipynb**

Our artificial neural network here would work great for classifying data into multiple (10) classes.

Here we use Softmax. Given a specific example Softmax will output probabilities (probabilities of that example being each class) where all of them add up to 1. Note that here we are assuming that each example belongs to exactly one class and can't have multiple labels (e.g. Dog and Frog in this case wouldn't make sense but for classifying the genre of different Netflix movies you may have one movie that is Action, Sci-Fi, and comedy). We would also have to use a different function other than softmax and different approach.

Although our images are fairly complicated. This might have been fine for numerical data and classifying that or simple images e.g. black and white letters or something.

Here this is going to require too much computation
It also is at risk of overfitting if there is too much variation and we can see that some of these cats for example, are in different positions and parts of the image and our network will struggle with this
As we are feeding in data flattened (each image is just an array of numeric values 1x 3072 vector) it literally treats all of the parts of the image the same and data which is close together (in 2D maybe they are together) the same as far apart data
Our images in 2D would be 32 x 32 x 3 size. 32 by 32 for dimensions height and width and the 3 for the colour depth (rgb, red green blue values like 120, -63, 24).

Flattened each image is 1 x 3072.

In the dataset folder "cifar-10-batches-py": there are data batch 1 to 5 and a test batch

data batch 1 to 5 are used for training and in total therefore there are 50000 images used for training
test batch has 10000 image
Each batch has 10000 images so e.g. data batch 1 for training has 10000 images

As a flattened image is 1 x 3072 vector the whole data matrix for data batch 1 (used for training) is (10000, 3072) We train this network on all 5 batches. Each batch it sees 10 times (thats why its 10 epochs and it starts at like low accuracy and works its way up). Then we test the network on the 1 testing batch using this unseen data to see if it can actually "see" and classify images correctly.

We get a low training accuracy and testing accuracy so we must change our approach.

## NN Convolutional Neural Network

This is better suited so we get higher accuracy on this complex image dataset than the flawed naive network.
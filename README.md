# INT2-Group-Project-CIFAR10

- [The CIFAR-10 Dataset Link](https://www.cs.toronto.edu/~kriz/cifar.html)

*Use this information in the report.*

## [**NN_Naive_ALLBatch_Refactored.ipynb**](https://github.com/UmerFakher/INT2-Group-Project-CIFAR10/blob/Automated-Dataset-DownloadExtractPicklePlot-UF/NN_Naive_ALLBatch_Refactored.ipynb)

Our artificial neural network here would work great for classifying data into multiple (10) classes.

Here we use Softmax. Given a specific example Softmax will output probabilities (probabilities of that example being each class) where all of them add up to 1. Note that here we are assuming that each example belongs to exactly one class and can't have multiple labels (e.g. Dog and Frog in this case wouldn't make sense but for classifying the genre of different Netflix movies you may have one movie that is Action, Sci-Fi, and comedy). We would also have to use a different function other than softmax and different approach.

Although our images are fairly complicated. This might have been fine for numerical data and classifying that or simple images e.g. black and white letters or something.

**Some drawbacks:**

* Here this is going to require too much computation

*It also is at risk of overfitting if there is too much variation and we can see that some of these cats for example, are in different positions and parts of the image and our network will struggle with this

* As we are feeding in data flattened (each image is just an array of numeric values 1x 3072 vector) it literally treats all of the parts of the image the same and data which is close together (in 2D maybe they are together) the same as far apart data
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

From looking at some of the images in this dataset you can easily pick the class/label of an image Dog or Frog etc. 
Your neurons look for specific features perhaps lines and then at a higher level of abstraction you may pick out shapes and then going even further you may pick out body parts, perhaps a cats head, tail and wiskers and you know its a cat.

lines -> shapes -> parts of image -> what image is made of

**We want to detect these features.** The small ones and put these together to get larger ones until we can work out what the image is.

So we need to apply small filters to the image to detects shapes/parts. These output **feature maps** which will highlight whether and where in the image the filters detect that part/shape/feature!

So this part is called **feature extraction** where we are getting lines and then shapes and then object parts detected from these feature maps, and then we must flatten all this information and run a **classification** similar to our network above. 

So essentially we are strapping on this idea of **convolutions** onto our network from before.

## ReLU activation function Quick explanation
We also used the ReLu activation function in the naive model as well.

[ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) literally is a activation function that given any input less than 0, it outputs 0 and any input greater than or equal to 0 it outputs that same value.

* ReLu helps making the model **nonlinear**
* It takes in a feature map as input and outputs the same but makes all the negative values 0 and leaves all positve values untouched

So ReLu is basically a flat line that suddenly spikes diagonally to the right after 0:

<a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">
  <img alt="ReLu Graph" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/ReLU_and_GELU.svg/1920px-ReLU_and_GELU.svg.png"
  width="300" height="300">
</a>

## [Pooling](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer)

**We said in the drawbacks** that there was just going to be too much computation involved with the naive network
so to deal with this we are going to use **pooling** to reduce the size

We can use max pooling to reduce the size of a feature map.

<a href="https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer">
  <img alt="ReLu Graph" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Max_pooling.png/471px-Max_pooling.png"
  width="300" height="300">
</a>
Max pooling with filter of dimensions 2 by 2 and stride as 2

Essentially we filter something like this:

1 0 2 3

4 6 6 8

3 1 1 0

1 2 2 4

Using outputting from a filter of 2x2 dimensions:

6 8 

3 4

So you can see it looks at the top left:

1 0

4 6   

and the max pooling makes these 4 values in to a single value which is 3 as it is the max

Then it pools top right then bottom left then bottom right

You can see here we reduced the size from 4 x 4 to 2 x 2 which is a pretty large saving when you scale.

Also a note here we assume we are using a *stride* of 2 which means that we go from

1 0         2 3

4 6   to    6 8

rather than using a stride of 1 which would move right by 1 step like this:

0 2

6 6

In which case it would reduce the whole thing from 4 x 4 to 3 x 3 using stride 1 with a filter of dimesion 2 x 2.


So this **pooling will help us** reduce these computations, dimensionality and overfitting as not as many numbers so this model should be better at dealing with differences and variations in images and placement of objects etc

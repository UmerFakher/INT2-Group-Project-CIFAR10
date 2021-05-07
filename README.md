# INT2-Group-Project-CIFAR10

- [The CIFAR-10 Dataset Link](https://www.cs.toronto.edu/~kriz/cifar.html)

## Model Summaries

Use this in report:

* CIFAR10_ANN_NaiveModel.ipynb -> Naive ANN Relu + softmax Classifier Unseen Acc of 41 %
* CIFAR10_CNN_Model_eachBatch10epoch.ipynb -> Simple CNN Feature Extraction Convolution & Pooling layers Unseen Acc of 66.7%
* CIFAR10_CNN_Model_concatBatches.ipynb -> CNN with combined training and less overfitting on batches (e.g. not 10 epochs of for each batch) Unseen Acc of 70.6%

Then needed to use Colab for vastly quicker training times and computing power:

* CIFAR10_CNN_Model_ColabReady.ipynb    -> Simple CNN Unseen Acc of 71 %
* CIFAR10_CNN_Model_ColabShortened.ipynb -> Larger CNN using Batch Normalisation + other Unseen Acc of 85% (usually 86%) 
* CIFAR10_CNN_Augmented_Model_Colab.ipynb -> Data Augmentation used. Unseen Acc of **87.87%.**
  *   **-> Saved Model File:** *"augmodel1_07-05-21_test_acc_87p.h5"* wDataAugmentation TestAcc 87.87%

## [Best Model CIFAR10_CNN_Augmented_Model_Colab.ipynb](https://github.com/UmerFakher/INT2-Group-Project-CIFAR10/blob/Automated-Dataset-DownloadExtractPicklePlot-UF/CIFAR10_CNN_Augmented_Model_Colab.ipynb)

Hardware and environment transitioned to  Google Colab Cloud based ipython notebook service to train the model on Cloud GPU in a more reasonable training time.

### Results for Augmented CNN Model:
![image](https://user-images.githubusercontent.com/66469756/117465858-04543400-af4a-11eb-8aca-64191b722509.png)

### Training time for Augmented CNN Model:
![image](https://user-images.githubusercontent.com/66469756/117465927-13d37d00-af4a-11eb-9193-9d2fecbabd0a.png)


### Confusion Matrix

![image](https://user-images.githubusercontent.com/66469756/117465501-ae7f8c00-af49-11eb-84e5-c544de0f512b.png)

### Example of a difficult problem 

However, it seems naturally that the network struggles with man-made vehicles due to feature similarities combined with the low resolution images. As seen in confusion matrix. Although, the model generally is pretty good.

![image](https://user-images.githubusercontent.com/66469756/117465562-bb03e480-af49-11eb-8477-9188266ec126.png)

### Model Diagram

![image](https://user-images.githubusercontent.com/66469756/117466109-43828500-af4a-11eb-8a23-a59ea10e92d2.png)

### Data Augmentation
![image](https://user-images.githubusercontent.com/66469756/117466263-6ca31580-af4a-11eb-8d12-f277827718e4.png)


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

E.g. [CIFAR10_CNN_Model_ColabShortened.ipynb](https://github.com/UmerFakher/INT2-Group-Project-CIFAR10/blob/Automated-Dataset-DownloadExtractPicklePlot-UF/CIFAR10_CNN_Model_ColabShortened.ipynb)

This is better suited so we get higher accuracy on this complex image dataset than the flawed naive network.

From looking at some of the images in this dataset you can easily pick the class/label of an image Dog or Frog etc. 
Your neurons look for specific features perhaps lines and then at a higher level of abstraction you may pick out shapes and then going even further you may pick out body parts, perhaps a cats head, tail and wiskers and you know its a cat.

lines -> shapes -> parts of image -> what image is made of

**We want to detect these features.** The small ones and put these together to get larger ones until we can work out what the image is.

So we need to apply small filters to the image to detects shapes/parts. These output **feature maps** which will highlight whether and where in the image the filters detect that part/shape/feature!

So this part is called **feature extraction** where we are getting lines and then shapes and then object parts detected from these feature maps, and then we must flatten all this information and run a **classification** similar to our network above. 

So essentially we are strapping on this idea of **convolutions** onto our network from before.

## [ReLU activation function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
We also used the ReLu activation function in the naive model as well.

[ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) literally is a activation function that given any input less than 0, it outputs 0 and any input greater than or equal to 0 it outputs that same value.

* ReLu helps making the model **nonlinear** and this is needed as our problem is complicated and multi-dimensional
* It takes in a feature map as input and outputs the same but makes all the negative values 0 and leaves all positve values untouched

So ReLu is basically a flat line that suddenly spikes diagonally to the right after 0:

<a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">
  <img alt="ReLu Graph" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/ReLU_and_GELU.svg/1920px-ReLU_and_GELU.svg.png"
  width="300" height="300">
</a>

Image Link: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

## [Pooling](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer)

**We said in the drawbacks** that there was just going to be too much computation involved with the naive network
so to deal with this we are going to use **pooling** to reduce the size

We can use max pooling to reduce the size of a feature map.

<a href="https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer">
  <img alt="ReLu Graph" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Max_pooling.png/471px-Max_pooling.png"
  width="300" height="300">
</a>

Image Link: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer

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


## Convolution Neural Network Summary

### Strength of our approach:

So an advantage of this convolutional approach is that not all nodes are connected to every other node like in a naive neural network, and as a result this helps to prevent **overfitting**. The advantage of pooling is that it reduces the dimensionality of our problem and consequently helps reduce overfitting. Moreover using **convolutions** and **pooling** will allow us to detect and extract features that could be located in different parts of the image, depending on the image example and this should mean in theory that the network will be more robust in handling for example a plane's wing shown in **different positions**. In summary, our model becomes more robust in the face of small differences and challenges that are presented by the varying dataset.

In addition, when the model learns the parameters for a filter, you can apply them in the in other parts of the image and as a result this reduces the number of weights needed in the network also.  This is often called parameter sharing: https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer.

As discussed above ReLu activation function helps make the model **nonlinear**, fit to solve our complicated and multi-dimensional problem. It is very efficient and quick to calculate as all it does is set all negative values to 0 and doesn't touch the positive values.

### Side note on potential improvement: 
As mention that the network will be more robust in handling for example a plane's wing shown in **different positions**, but just as a clarification if we have complex rotations of features or even different scaled images such as smaller cats and larger cats then the convolution neural network will require such training examples that have been scaled or rotated. 

Luckily it seems the Cifar-10 dataset had quite a different images like this so this should be sufficient however if we are still reaching a low testing accuracy then this could be an area to improve perhaps using **data augmentation** where we could create new images so we have more training examples from existing ones that are scaled or rotated.

### Quick Summary

The convolutional neural network consists of:
* Feature Extraction ('Convolution with ReLu, then pooling' which may be repeated several times) 
Each convolution with ReLu layer learns a feature through filters (a bunch of weights), for example the first convolution with relu layer may learn lines and shapes whereas the next could learn specific object parts such as plane wing or cat's tail and this builds up to more abstract features.

The network automatically detects and learns the filters as part of training we input multiple instances of images (for all 10 classes) for example that are classed as planes or cats. Using backpropagation the network will figure out the good numerical values of the weights in filters. Hyperparameters may include how many filters we want to specify and the dimensions of these filters. The values of the filters are then self-learned by the network through training.

* Softmax Classification after a flatten, to generate probabilities of which image class, the image example that is given as input, could belong to. This layer ends up classifying what the image is out of our 10 classes.


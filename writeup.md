# German Traffic Sign Recognition

[//]: # (Image References)

[dataset-sample]: ./writeup-assets/dataset-sample.png "Dataset Sample"
[class-distribution]: ./writeup-assets/class-distribution.png "Class Distribution"
[preprocessing-comparison]: ./writeup-assets/preprocessing-comparison.png "Preprocessing Comparison"
[lenet-5]: ./writeup-assets/lenet-5.png "LeNet-5"
[new-images]: ./writeup-assets/new-images.png "New Images"
[new-images-p]: ./writeup-assets/new-images-p.png "New Images Processed"
[new-predictions]: ./writeup-assets/new-predictions.png "New Image Predictions"
[conv-layer-1]: ./writeup-assets/conv-layer-1.png "Layer 1"
[conv-layer-2]: ./writeup-assets/conv-layer-2.png "Layer 2"
[conv-layer-3]: ./writeup-assets/conv-layer-3.png "Layer 3"
[conv-layer-4]: ./writeup-assets/conv-layer-4.png "Layer 4"


**The goals / steps of this project are the following**

1. Load, Explore, summarize and visualize the data set
2. Design, train and test a model architecture
3. Use the model to make predictions on new images & analyze the results
4. Visualize the internal state of the model


## 1. Load, explore, summarize and visualize the data set

I used the German Traffic Sign Dataset. It is already pickled, and can be
downloaded with [this link](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip).
In total, it has 59,839 images with 43 different classes, and is divided
into a training set, validation set and test set with 34,799, 12,630 and
4,410 samples respectively. All the images are 32x32 and have three color
channels, producing a consistent (32,32,3) shape.

The labels are indexed values ranging from 0-42, each representing a
unique class, and there is an accompanying signnames.csv file to map each
value to a sign name.

Below is a labelled sample representing each class in the dataset.

![dataset-sample]

As the chart shows below, the classes are not evenly represented in the
dataset, but the training, validation and test sets are proportional on
a class-by-class basis.

![class-distribution]

## 2. Design, train and test a model architecture


### Additional data creation

I augmented the training dataset with additional images, rotated +15 and
-15 degrees for every image to add more information about orientation
to improve translational invariance.  This tripled the size of the training
set.  More could definitely be done.  I could add more data by via scaling,
translating, brightness adjustments, etc.

### Preprocessing techniques

I started with converting the images to grayscale because of the successful
results it produced for Pierre Sermanet and Yann LeCun in their paper
[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
The results with the model I was using at the time (LeNet-5) were decent,
but I knew it could get better. I noticed many images had poor lighting
and lacked contrast.


Curious if different techniques would yield better results, I started to
research and compare different approaches.

![preprocessing-comparison]

Histogram Equalization and it's more sophisticated cousin, Contrast-Limited
Adaptive Histogram Equalization (CLAHE), showed promising results.  After
testing them out (and somewhat to my surprise), basic Histogram Equalization
proved to be better suited in this case.  This could be a topic for future
research for me.

After histogram equalization, I normalized the dataset to zero mean, and
expanded the dimensions to include 1 color channel, for a final shape of
(x, 32, 32, 1), where x is the number of images in the set. Finally, I
shuffled the training images and labels.

To sum it up, the data preparation steps I used are:

1. Add +/- 15deg rotational variants
2. Histogram Equalization
3. Zero-mean normalization
4. Dimensional expansion to 4D shape
5. Shuffle


### Model architecture & Hyper parameters

I started with an adapted implementation of LeNet-5.

![lenet-5]

LeNet-5 worked well, but It didn't hit the accuracy I was looking for, so
I began a process of iterative improvements.  I decided to add Dropout to
help prevent over-fitting, and to make the model larger and deeper so that
it might capture more insight from the training set.  I also chose to
incorporate max pooling to keep activation signals strong, while still
achieving dimensionality reduction.

Each adjustment was tested to see if it actually provided positive results.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image 						|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x6 	|
| Max pooling	      	| 2x2 stride, valid padding, outputs 15x15x6     				|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 11x11x16 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 11x11x32 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 11x11x32 	|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x64		|
| RELU					|												|
| Flatten       	    | outputs 1600 									|
| Fully connected		| outputs 1600 									|
| Dropout       		| 50% keep probability							|
| Fully connected		| outputs 192									|
| Dropout       		| 50% keep probability							|
| Fully connected		| outputs 43 logits								|

The output logits are then passed through a softmax function.  Cross entropy
and a loss operation is calculated, which is then minimized with the Adam
optimizer.  Meanwhile, the labels are one-hot encoded.

**Hyper Perameters**

- Epochs: 15
- Batch Size: 512
- Learning Rate: 0.001
- Keep Probability: 0.50


### Results

My final model results were:
* Training set accuracy: 1.000
* Validation set accuracy: 0.973
* Test set accuracy of 0.951

## 3. Use the model to make predictions on new images & analyze the results

I chose ten new images to test the trained model against:

![new-images]

After applying the above mentioned preprocessing and normalization:

![new-images-p]

The prediction results were quite good.  The model correctly identified
all 10 of the new images with 100% accuracy!

The top 5 softmax probabilities for each new image are displayed below:

![new-predictions]

### Wow, 100%? Not so fast!

The accuracy is quite high if the images are carefully cropped to squares
that are similar to the images on the dataset.  The model doesn't
generalize well enough to handle images that aren't so carefully prepared.
Improvements could be made by adding more augmented samples to the dataset
so that the model could train on more images that may appear to be less
than ideal.

## 4. Visualize the internal state of the model

Using feature maps, we can actually peer into the inner state of a
Convolutional Neural Network, and visualize what features the network is
noticing at different layers in the network.  As you will see, the top
layers pick up on larger object, but as the network goes deeper, the
activations get more and more granular.

Below are some examples of what the network sees at each convolutional
layer:

![conv-layer-1]

![conv-layer-2]

![conv-layer-3]

![conv-layer-4]
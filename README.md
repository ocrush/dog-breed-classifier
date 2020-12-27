[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/dog_web_app.png "Web App"
[image3]: ./images/CNN.png "Sample CNN"
[image4]: ./images/model_summary.png "Model Summary"

## Project Overview
This project uses Convolutional Neural Networks (CNNs) to classify breeds of dogs.  We will build a pipeline to process real-world, user-supplied images.  Given an image of a dog, our algorithm will identify an estimate of the cannie's breed.  If supplied an image of a human, the code will identify the resembling dog breed.

This project is an implementation of the  [Dog Classifier Project](https://github.com/udacity/dog-project.git).    

## Problem Statement 

How best to identify dog breeds from their images?  This is not a easy task.  Even for humans, it is hard to accurately classify certain breeds of dogs as they look very much alike.  So, we are not expecting to achieve high accuracy for our model but over 60% would be acceptable.

CNN's have proven very effective in areas such as image recognition and classification.  For example, identifying faces, traffic signs, and even used in robotic vision.  CNN's use several different layers.  The basic idea is to extract important features from the image.  The Convolution operation captures the local dependencies in the original image by using different type of filters such as edge detection, box blur, etc, takes care of non-linearity using methods such as ReLU  and uses down sampling such as max pooling to reduce the dimensionality of each feature map.  
![Sample Output][image3]

Project will present the users with a web app that allows users to upload their own dog or human images and their matching or resembling breed image will be displayed back to the user.  For this, we will need to recognize dog and humans in the image itself.  We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  For dogs detection, we will use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.

Next, we will compare and contrast model that will be written from scratch using Keras and a model that uses Transfer learning.  Winner of this comparison will be used to the breed from the dog image. 

## Project Instructions

### Instructions
1. pip install -r requirements.txt
2. Clone the repository, navigate to the downloaded folder, and run the web app.  If there are dependency issue, please follow steps 2 and beyond.
```	
git clone https://github.com/ocrush/dog-breed-classifier.git
cd dog-breed-classifier/app
python run.py
Use web browser and go to http://0.0.0.0:3001/
```
    
## Metrics
We will mostly use accuracy to evaluate the performance of the model.  From a given set of data, how many human faces and dog images are we correctly identifying.  From a given set of data, how many breeds are we able to correctly identify.

## Data Exploration
[dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) was provided by Udacity.

There are 133 total dog categories.
There are 8351 total dog images.

There are 6680 training dog images.
There are 835 validation dog images.
There are 836 test dog images.

## Data Visualization
Since the data was presented in images, manually viewed several image files to get an idea of what type of images of dogs were going to be used for training the model.  There are some breeds that have more images for training than others. 
   
## Data Preprocessing
[dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) was provided by Udacity and it was already partitioned into train, test, and validation.  Since we are using Tensorflow backend some of teh data needed to be formatted properly to be used with Tensorflow.  See the Python notebook for more info. 

## Methods and Models

To only support dogs and human faces, code needed to identify if there is a human or dog available in the image.  

* OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images
* A pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images
* In case no dog or human is present, code will prompt user with an error

Next, several different architectures were constructed for CNN algorithm from scratch.  Please see the Python notebook for more details.  Experimentation was done with having different depth, filters, size of the filter matrix, use of fully connected layers, etc.  The final model that was built from scratch resulted in 23.80% accuracy.  Acceptable accuracy here was anything over 1%.  So, good results.
![Model Summary][image4]   

Here a depth of 4 is used with filters increasing fom 16, 32, 64, and 128.  Max pooling is used to down sample after each convolution step with global averaging and dropout before feeding the output to two fully connected layers which use deep learning to learn from the reduced feature set.

Even though the accuracy was 23%, it is most likely not usable in real world.

Transfer Learning  
## Results
![Sample Output][image2]




2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. For required Python packages refer to this [README](https://github.com/udacity/dog-project/blob/master/README.md)

#### Notebook
dog-app.ipynb - This notebook was used to train/test the CNN network in Udacity's workspace
1. Please follow the [instructions](https://github.com/udacity/dog-project/blob/master/README.md) provided in Udacity's README.
2. Personally, it was easy to train/test the CNN model using the Udacity workspace.

### Resources
The following articles were used to better understand CNN and implement the final architecture:

1. https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
2. https://victorzhou.com/blog/keras-cnn-tutorial/
3. https://keras.io/examples/vision/mnist_convnet/
4. https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
5. https://roytuts.com/upload-and-display-image-using-python-flask/

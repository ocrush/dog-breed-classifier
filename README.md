[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/dog_web_app.png "Web App"


## Project Overview
This project is an implementation of the  [Dog Classifier Project](https://github.com/udacity/dog-project.git).  A fun project that will utilize a web app to classify dog breeds.  Users can upload images of dogs or people and the web app will figure out the matching breed for the dogs but in case of humans, it will find the resembling dog breed.  


![Sample Output][image2]


## Project Instructions

### Instructions

1. Clone the repository, navigate to the downloaded folder, and run the web app.  If there are dependency issue, please follow steps 2 and beyond.
```	
git clone https://github.com/ocrush/dog-breed-classifier.git
cd dog-breed-classifier/app
python run.py
Use web browser and go to http://0.0.0.0:3001/
```
1. For web app, you can use the requirements.txt file to install required Python Packages
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

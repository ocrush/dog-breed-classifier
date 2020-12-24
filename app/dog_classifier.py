from glob import glob
import numpy as np
from sklearn.datasets import load_files
from keras.utils import np_utils
from flask import Flask, flash, request, redirect, url_for, render_template
from keras import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from extract_bottleneck_features import *
from tqdm import tqdm
import cv2
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50


# Flask gave errors while using ResNet50 for dog detection.  Reading online suggests using a graph object after
# loading model
class ResNet50Wrap:
    '''
    Just wrapper for Resnet50 so that we can have the model prediction under tf.graph
    otherwise same as in the notebook
    '''
    def __init__(self):
        self.model = ResNet50(weights='imagenet')
        self.graph = tf.get_default_graph()

    def predict_labels(self, img_f):
        # returns prediction vector for image located at img_path
        img = preprocess_input(img_f)
        with self.graph.as_default():
            return np.argmax(self.model.predict(img))


class DogClassifier:
    '''
    Class to predict the dog breed
    Creates a model, loads the saved weights.  These weights were result of running the notebook
    Mostly used for the web app
    '''
    def __init__(self):
        self.model = Sequential()
        self.model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        self.model.add(Dense(133, activation='softmax'))
        self.model.load_weights('../saved_models/weights.best.Resnet50.hdf5')
        self.graph = tf.get_default_graph()
        self.dog_names = [item[20:-1] for item in sorted(glob("../dogImages/train/*/"))]

    def predict_breed(self, features):

        # obtain predicted vector
        with self.graph.as_default():
            bottleneck_feature = extract_Resnet50(features)
            predicted_vector = self.model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return self.dog_names[np.argmax(predicted_vector)]

class DogPredictor:
    '''
    Main class called by run.py to get the predicted breed
    Most of the functions here were lifted from the Python notebook

    '''
    def __init__(self):
        # create model
        # define ResNet50 model
        self.ResNet50_model = ResNet50Wrap()
        # extract pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

        self.dog_classifier = DogClassifier()
        #self.train_files, _ = self.load_dataset('../dogImages/train')

    # define function to load train, test, and validation datasets

    def load_dataset(self, path):
        data = load_files(path)
        dog_files = np.array(data['filenames'])
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
        return dog_files, dog_targets

    def get_matching_dog_file(self, dog_files, predicted_dog_str):
        '''
        Not used - It was returning a matching file but now rep images are stored in static/images
        :param dog_files:
        :param predicted_dog_str:
        :return:
        '''
        for i, s in enumerate(dog_files):
            if predicted_dog_str in s:
                return s
        return ''

    def path_to_tensor(self, img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def paths_to_tensor(self, img_paths):
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

    # returns "True" if face is detected in image stored at img_path
    def face_detector(self, img_path):
        '''

        :param img_path: path for the image file
        :return: returns true if image has a human face
        '''
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def dog_detector(self, img_path):
        '''

        :param img_path: path of the image file
        :return: if dog is in the image
        '''
        prediction = self.ResNet50_model.predict_labels(self.path_to_tensor(img_path))
        return ((prediction <= 268) & (prediction >= 151))

    def dog_breed_matching(self, img_path):
        '''
        Find the matching breed for the image.  In case image does not contain human or dog, results in an error
        :param img_path: path of uploaded image
        :return: tuple.  Matching breed and error.  In case of error, matching breed is set to empty and vice versa
        '''
        #print("Image path = {}".format(img_path))
        error = "Error: Only dog and human picture are supported at this time"
        if self.dog_detector(img_path) or self.face_detector(img_path):
            # extract bottleneck features

            predicted_result = self.dog_classifier.predict_breed(self.path_to_tensor(img_path))
            predicted_breed = predicted_result.split('.')[1]
            print("This photo look like {}".format(predicted_breed))
            #matching_dog_file = self.get_matching_dog_file(self.train_files, predicted_result)
            if len(predicted_breed) > 0:
                error = ''
                return predicted_breed, error

        return '', error

if __name__ == '__main__':
    doc = DogPredictor()
    matching_file,error = doc.dog_breed_matching("static/uploads/Boxer_02360.jpg")
    print(matching_file)

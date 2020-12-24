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

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def ResNet50_predict_labels(img_path):
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    from keras.applications.resnet50 import ResNet50

    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def get_matching_dog_file(dog_files, predicted_dog_str):
    for i, s in enumerate(dog_files):
        if predicted_dog_str in s:
            return s
    return ''


class DogClassifier:

    def __init__(self):
        # create model
        Resnet50_model = Sequential()
        Resnet50_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        Resnet50_model.add(Dense(133, activation='softmax'))
        Resnet50_model.load_weights('../saved_models/weights.best.Resnet50.hdf5')
        self.model = Resnet50_model
        self.dog_names = [item[20:-1] for item in sorted(glob("../dogImages/train/*/"))]
        self.train_files, _ = load_dataset('../dogImages/train')

    # define function to load train, test, and validation datasets

    def predict_breed(self, img_path):
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = self.model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return self.dog_names[np.argmax(predicted_vector)]

    def dog_breed_matching_file(self, img_path):
        error = "Error: Only dog and human picture are supported at this time"
        if dog_detector(img_path) or face_detector(img_path):
            predicted_result = self.predict_breed(img_path)
            predicted_breed = predicted_result.split('.')[1]
            print("This photo look like {}".format(predicted_breed))
            matching_dog_file = get_matching_dog_file(self.train_files, predicted_result)
            if len(matching_dog_file) > 0:
                error = ''
                return matching_dog_file, error

        return '', error

doc = DogClassifier()
matching_file,error = doc.dog_breed_matching_file("./static/uploads/Afghan_hound_00081.jpg")
print(matching_file)

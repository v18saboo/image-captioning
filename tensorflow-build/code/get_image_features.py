from tensorflow.contrib import keras
from keras.applications.vgg16 import VGG16
import keras.preprocessing
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import os

def features(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    fc2_features = model.predict(x)
    return fc2_features


if __name__ == "__main__":
	directory = './data/test_images/'
	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
	files = os.listdir(directory)
	img_output = np.zeros((1,4096))
	for i in files:
	    img_output = np.vstack((img_output,features(directory+i)))
	np.save('./data/test_features',img_output[1:])    
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np


class ImageModel:
    def __init__(self,base_model,extract_layer):
        if(base_model == 'VGG19'):
            from keras.applications.vgg19 import preprocess_input
            self.base_model = keras.applications.vgg19.VGG19(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(extract_layer).output)
            self.target_size = (224,224)
            
        else if(base_model == 'ResNet'):
            from keras.applications.resnet50 import preprocess_input, decode_predictions
            self.base_model = keras.applications.resnet50.ResNet50(weights='imagenet') #Add outputs argument.
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(extract_layer).output)
            self.target_size = (224,224)
            
        else if(base_model == 'InceptionResNetV2'):
            self.base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',
                                                                     input_tensor=None, input_shape=None,
                                                                     pooling=None, classes=1000) #Add outputs argument.
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(extract_layer).output)
            self.target_size = (299,299)
            
    def get_model(self):
        return self.model
    
    def predict(self,img_path):      
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        intermediate_features = model.predict(x)
        

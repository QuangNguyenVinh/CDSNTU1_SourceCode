import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class SignClassification:
    @staticmethod
    def loadmodel(path):
        return load_model(path)

    def __init__(self, path):
        self.model = self.loadmodel(path)
        self.graph = tf.get_default_graph()
        self.IMG_SIZE = 48
        self.isSign = 0.91
        self.label = {0 : "go_straight",
                     1 : "turn_left",
                     2 : "turn_right",
                     3 : "no_turn_left",
                     4 : "no_turn_right",
                     5 : "stop",
                    None: "None"}

    def getModel(self):
        return self.model

    def predict(self, X):
        with self.graph.as_default():
            return self.model.predict(X)

    def getLabel(self, src): #Input: Gray Image
        img = cv2.resize(src, (self.IMG_SIZE, self.IMG_SIZE))
        img = img.reshape((-1,self.IMG_SIZE, self.IMG_SIZE, 1))
        result = self.model.predict(img)
        #print("Label and Score: ",np.argmax(result), result[0][np.argmax(result)])
        if result[0][np.argmax(result)] >= self.isSign:
            print(result[0][np.argmax(result)])
            return np.argmax(result)
        else:
            return None





signClassify = SignClassification('model/CNN_perfect.h5')
img = np.zeros((signClassify.IMG_SIZE, signClassify.IMG_SIZE, 1))
signClassify.predict(img.reshape((-1,signClassify.IMG_SIZE, signClassify.IMG_SIZE, 1)))
print("SIGN CLASSIFICATION READY")


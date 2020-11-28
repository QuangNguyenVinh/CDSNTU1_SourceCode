import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class loadModel:
    @staticmethod
    def loadmodel(path):
        return load_model(path)

    def __init__(self, path):
       self.model = self.loadmodel(path)
       self.graph = tf.get_default_graph()
    def getModel(self):
        return self.model

    def predict(self, X):
        with self.graph.as_default():
            return self.model.predict(X)


lane_model = loadModel('model/model.h5')
lane_model.getModel().summary()

def Advance_lane(img):
    global lane_model
    img = cv2.resize(img, (240, 160))
    predict = lane_model.predict(np.array([img/255.0]))[0]
    center = int(predict[0]*240)
    return center

frame = np.zeros((480, 640, 3))
center = Advance_lane(frame)
print("ADVANCE LANE READY")
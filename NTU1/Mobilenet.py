import cv2
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import applications
from tensorflow.keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import cv2
import numpy as np
import time
mbl = applications.mobilenet.MobileNet(weights=None, include_top=False, input_shape=(160,320,3))
x = mbl.output

model_tmp =  Model(inputs = mbl.input, outputs = x)
layer5, layer8, layer13 = model_tmp.get_layer('conv_pw_5_relu').output, model_tmp.get_layer('conv_pw_8_relu').output, model_tmp.get_layer('conv_pw_13_relu').output

fcn14 = Conv2D(filters=2 , kernel_size=1, name='fcn14')(layer13)
fcn15 = Conv2DTranspose(filters=layer8.get_shape().as_list()[-1] , kernel_size=4, strides=2, padding='same', name='fcn15')(fcn14)
fcn15_skip_connected = Add(name="fcn15_plus_vgg_layer8")([fcn15, layer8])
fcn16 = Conv2DTranspose(filters=layer5.get_shape().as_list()[-1], kernel_size=4, strides=2, padding='same', name="fcn16_conv2d")(fcn15_skip_connected)
# Add skip connection
fcn16_skip_connected = Add(name="fcn16_plus_vgg_layer5")([fcn16, layer5])
# Upsample again
fcn17 = Conv2DTranspose(filters=3, kernel_size=16, strides=(8, 8), padding='same', name="fcn17", activation="softmax")(fcn16_skip_connected)

model = Model(inputs = mbl.input, outputs = fcn17)
model.load_weights('model/Mobilenet_RoadLine.h5')
img = np.zeros((160, 320, 3))
model.predict(np.expand_dims(img, axis=0))
cap = cv2.VideoCapture(r'D:\\CDS\\DiRa_CDSNTU1\\PyCDS\\output.avi')



while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    frame = frame[60:380]
    frame = cv2.resize(frame, (320, 160), interpolation=cv2.INTER_NEAREST)
    # Display the resulting frame

    my_preds = model.predict(np.expand_dims(frame,axis = 0))
    #First decode
    road = my_preds[0,:,:,1]
    line = my_preds[0,:,:,2]
    segment = np.zeros((160, 320, 3))
    segment[road >= 0.5] = (0,0,255)
    segment[line >= 0.5] = (0,255,0)
    cv2.imshow('seg', segment)

    #Second decode
    #my_preds = np.where(my_preds >= 0.5, 1, 0)
    #img = my_preds.reshape(160, 320, 3)
    #img = img * 255. *1.
    #cv2.imshow('seg', img)

    cv2.imshow('rgb', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
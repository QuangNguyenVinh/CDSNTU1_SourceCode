from Classic import classic
import cv2
import time
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from DetectHorizontalLine import *
cap = cv2.VideoCapture(r'output4.avi')

height, width = 160, 320
input_img = Input((height, width, 3), name='img')

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)

u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c4)
u5 = concatenate([u5, c3])
c6 = Conv2D(32, (3, 3), activation='relu', padding='same') (u5)
c6 = Conv2D(32, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c2])
c7 = Conv2D(16, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(16, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c1])
c8 = Conv2D(8, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(8, (3, 3), activation='relu', padding='same') (c8)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

model = Model(inputs=[input_img], outputs=[outputs])
model.load_weights('model/line_2019.h5')
img = np.zeros((height, width, 3))
model.predict(np.expand_dims(img, axis=0))
img2 = cv2.imread(r'D:\\OpenSource\\DataFull\\snow_shadow\\5\\4780.jpg')

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    frame = frame[60:380]
    frame = cv2.resize(frame, (320, 160), interpolation=cv2.INTER_NEAREST)
    #frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
    #frame = frame[30:190]
    # Display the resulting frame
    frame = img2.copy()
    frame = frame[50:210]

    my_preds = model.predict(np.expand_dims(frame,axis = 0))

    my_preds = np.where(my_preds >= 0.5, 1, 0)
    img = my_preds.reshape(height, width)
    img = img*255.*1.

    cv2.imshow('seg', img)

    #res = np.expand_dims(frame / 255., axis=0)
    #cv2.imshow('seg', )
    #cv2.imshow('seg', bin * 255.)
    cv2.imshow('rgb', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
import cv2
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import applications
from tensorflow.keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
import cv2
import numpy as np
import time
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2)))
K.set_session(sess)
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


IMAGE_H = 160
IMAGE_W = 320

src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[135, IMAGE_H], [185, IMAGE_H], [0 - 20, 0], [IMAGE_W + 20, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
def get_bird_view(img):

    img = img[80:(80+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    return warped_img

fps = 0
count = 0
out = cv2.VideoWriter('seg2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (IMAGE_W,IMAGE_H))
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    start = time.time()
    frame = frame[60:380]
    frame = cv2.resize(frame, (320, 160), interpolation=cv2.INTER_NEAREST)


    my_preds = model.predict(np.expand_dims(frame,axis = 0))
    #First decode
    road = my_preds[0,:,:,1]
    line = my_preds[0,:,:,2]
    segment = np.zeros((160, 320, 3))
    segment[road >= 0.5] = (0,0,255)
    segment[line >= 0.5] = (0,255,0)
    #segment = cv2.resize(segment, (320, 240))
    #segment = get_bird_view(segment)
    segment = np.uint8(segment)
    #out.write(segment)

    #Second decode
    my_preds = np.where(my_preds >= 0.5, 1, 0)
    img = my_preds.reshape(160, 320, 3)
    img = img * 255. *1.
    cv2.imshow('seg2', img)
    #img = np.uint8(img)
    #out.write(img)

    print("Delta time: ", time.time() - start)

    cv2.imshow('rgb', frame)
    cv2.imshow('seg1', segment)
    fps += 1 / (time.time() - start)
    count += 1
    if count == 30:
        print("FPS: ",fps / float(count))
        count = 0
        fps = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()

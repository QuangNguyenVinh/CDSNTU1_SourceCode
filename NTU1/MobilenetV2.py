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
K.set_learning_phase(0)
mbl = applications.mobilenet_v2.MobileNetV2(weights=None, include_top=False, input_shape=(160, 320, 3))
x = mbl.output
model_tmp = Model(inputs=mbl.input, outputs=x)
layer4, layer9, layer17 = model_tmp.get_layer('block_4_add').output, model_tmp.get_layer(
    'block_9_add').output, model_tmp.get_layer('out_relu').output
fcn18 = Conv2D(filters=2, kernel_size=1, name='fcn18')(layer17)
fcn19 = Conv2DTranspose(filters=layer9.get_shape().as_list()[-1], kernel_size=4, strides=2, padding='same',
                        name='fcn19')(fcn18)

fcn19_skip_connected = Add(name="fcn19_plus_vgg_layer9")([fcn19, layer9])
fcn20 = Conv2DTranspose(filters=layer4.get_shape().as_list()[-1], kernel_size=4, strides=2, padding='same',
                        name="fcn20_conv2d")(fcn19_skip_connected)

# Add skip connection
fcn20_skip_connected = Add(name="fcn20_plus_vgg_layer4")([fcn20, layer4])
# Upsample again
fcn21 = Conv2DTranspose(filters=2, kernel_size=16, strides=(8, 8), padding='same', name="fcn21", activation="softmax")(
    fcn20_skip_connected)
model1 = Model(inputs=mbl.input, outputs=fcn21)
#model1.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model1.summary()
model1.load_weights(r'D:\\CDS\\DiRa_CDSNTU1\\PyCDS\\model\\chungket\\model-mobilenetv2-round4.h5')

cap = cv2.VideoCapture(r'D:\\CDS\\DiRa_CDSNTU1\\PyCDS\\output.avi')
fps = 0
count = 0
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    start = time.time()
    frame = cv2.resize(frame[:500, :], (320, 160))
    cv2.imshow("a",((frame/255.)*255.).astype('uint8'))
    res = model1.predict(np.expand_dims(frame/255., axis=0))

    # seg = np.argmax(res, axis=3)[0]
    seg = np.argmax(res, axis=3)[0]
    #frame[seg == 1] = [0, 0, 255]
    cv2.imshow('rgb', frame)
    #print("Delta time: ", time.time() - start)

    fps += 1 / (time.time() - start)
    count += 1
    if count == 30:
        print("FPS: ",fps / float(count))
        count = 0
        fps = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

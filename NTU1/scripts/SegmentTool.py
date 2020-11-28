import cv2
import time
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate



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
model.load_weights('model/line.h5')
img = np.zeros((height, width, 3))
model.predict(np.expand_dims(img, axis=0))

DST_FOLDER = r'D:\\OpenSource\\lane_mask\\'
SRC_FOLDER = r'D:\\OpenSource\\DataFull\\DataFull1\\'
id_lane = [34, 34, 34]
id_road = [7, 7, 7]
x_offset = 0
y_offset = 50
for index in range(0,6969):
    img = cv2.imread(SRC_FOLDER + str(index) + ".jpg")
    blank_image = np.zeros(shape=[160, 320, 3], dtype=np.uint8)
    blank_image_2 = np.zeros(shape=[240, 320, 3], dtype=np.uint8)
    frame = img[50:210]
    my_preds = model.predict(np.expand_dims(frame, axis=0))
    my_preds = np.where(my_preds >= 0.5, 1, 0)
    img = my_preds.reshape(height, width)
    img = img * 255. * 1.
    img = np.uint8(img)
    blank_image[np.where(img == 255)[0], np.where(img == 255)[1]] = id_lane

    blank_image_2[y_offset:y_offset + blank_image.shape[0], x_offset:x_offset + blank_image.shape[1]] = blank_image
    #cv2.imwrite(DST_FOLDER + str(index) + "_mask.png", blank_image_2)
    print("STT: " + str(index))
    cv2.waitKey(5)
print("DONE")

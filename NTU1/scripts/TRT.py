import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.platform import gfile
from keras import backend as K
K.set_learning_phase(0)
class Model():
    def __init__(self, path):
        self.graph = tf.Graph()
        sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1)))
        self.sess = sess
        with self.graph.as_default():
            trt_graph = self.read_pb_graph(path + 'tensorRT_FP32.pb')
            tf.import_graph_def(trt_graph, name='')
            self.input = self.sess.graph.get_tensor_by_name('input_1:0')
            self.output = self.sess.graph.get_tensor_by_name('road_output/truediv:0')
            self.predict(np.zeros((160, 320, 3)))

    def read_pb_graph(self, path):
        with gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def
    def predict(self, image):
        image = np.expand_dims(image, 0)
        pred = self.sess.run(self.output, feed_dict={self.input: image/255.})
        pred = np.argmax(pred, axis=3)[0]
        return pred

def get_left_right_points(seg, box_size = 5, y = 60):
    left = 0
    right = 320
    p_left = (-1,-1)
    hasLeft = False
    p_right = (-1,-1)
    hasRight = False
    while(left < 320):
        if np.sum(seg[y - box_size: y + box_size, max(0,left - box_size) : min(left + box_size,320)]) == (box_size*2) ** 2:
            p_left = (left + 3, y) #y,x
            hasLeft = True
            break
        left += 1
    while(right > 0):
        if np.sum(seg[y - box_size: y + box_size, max(0, right - box_size): min(right + box_size, 320)]) == (box_size*2) ** 2:
            p_right = (right - 3, y) # y,x
            hasRight = True
            break
        right -= 1
    if(hasLeft == False and hasRight == False):
        return ()
    elif(hasLeft == True or hasRight == True) and p_left == p_right:
        return  p_left,p_right,-1
    return p_left, p_right
def get_center_each_side(p_left, p_right):
    p_center = (p_left[0] + p_right[0])//2, (p_left[1] + p_right[1])//2
    p_center_left = ((p_center[0] + p_left[0])//2, p_left[1])
    p_center_right = ((p_center[0] + p_right[0])//2, p_right[1])
    return p_center, p_center_left, p_center_right

m = Model(r'D:\\CDS\\DiRa_CDSNTU1\\PyCDS\\model\\chungket\\')
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
    seg = m.predict(frame)
    frame[seg == 1] = [0,0,255]
    p = get_left_right_points(seg)
    if(len(p) == 2):
        #print("Point left {} and right {}".format(p[0], p[1]))
        cv2.circle(frame, p[0], 5, (255,0,0))
        cv2.circle(frame, p[1], 5, (0,255,0))

        x,y,z = get_center_each_side(p[0], p[1])
        cv2.circle(frame, x, 15, (255,255,255))
        cv2.circle(frame, y, 15, (255,255,255))
        cv2.circle(frame, z, 15, (255,255,255))

        print("Angle: {}".format(z[0] - 160))


    elif(len(p) == 3):
        #Mode 3
        cv2.circle(frame, p[1], 10, (255, 0, 0))
        print("Mode 3 ON Point left {} and right {}".format(p[0], p[1]))
    elif(len(p) == 0):
        pass

    cv2.imshow('rgb', frame)

    fps += 1 / (time.time() - start)
    count += 1
    if count == 30:
        print("FPS: ",fps / float(count))
        count = 0
        fps = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

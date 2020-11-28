import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import time
import cv2
tf.keras.backend.set_learning_phase(0)


class Model:
    def __init__(self, path=None):
        if path is None:
            #path = Config().model_lane_path
            path = None

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph,
                               config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.15)))

        with self.graph.as_default():
            trt_graph = self.read_pb_graph(path)
            tf.import_graph_def(trt_graph, name='')
            self.input = self.sess.graph.get_tensor_by_name('input_1:0')
            self.output = self.sess.graph.get_tensor_by_name('fcn21/truediv:0')
            self.feature = self.sess.graph.get_tensor_by_name('fcn18/my_trt_op_35:0')
            self.predict(np.zeros((160, 320, 3)))

    def read_pb_graph(self, path):
        with gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        return graph_def

    def predict(self, image):
        image = np.expand_dims(image, 0)
        pred, f = self.sess.run([self.output, self.feature], feed_dict={self.input: image / 255.})
        pred = np.argmax(pred, axis=3)[0]

        return pred
m = Model(r'D:\\CDS\\DiRa_CDSNTU1\\PyCDS\\model\\chungket\\lane.pb')
cap = cv2.VideoCapture(r'D:\\CDS\\DiRa_CDSNTU1\\PyCDS\\output_phai.avi')
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
    cv2.imshow('rgb', frame)

    fps += 1 / (time.time() - start)
    count += 1
    if count == 30:
        print("FPS: ", fps / float(count))
        count = 0
        fps = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
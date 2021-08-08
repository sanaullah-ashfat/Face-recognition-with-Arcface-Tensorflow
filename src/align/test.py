import cv2
import os
import tensorflow as tf
from src.align import detect_face


class DETECTION:

    def __init__(self):
        self.keypoint =None
        self. path = os.getcwd()

    def landmarks(self,point):
        keypoints= None
       
        for bounding_box in (point.T):
            keypoints=[(((int(bounding_box[0]), int(bounding_box[5]))),
                        ((int(bounding_box[1]), int(bounding_box[6]))),
                        ((int(bounding_box[2]), int(bounding_box[7]))),
                        ((int(bounding_box[3]), int(bounding_box[8]))),
                        ((int(bounding_box[4]), int(bounding_box[9]))))]
    
        return keypoints
    

    def detect(self,img):

        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, self. path+"/src/align/input/")
        
        minsize = 20 # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor
        bounding_boxes, point = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        landmark =self.landmarks(point)

        return bounding_boxes, landmark


import os
import cv2
import time
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from src.modules import utils
from src.ALIGNMENT import ALIGN
from src.modules.utils import l2_norm
from src.modules.models import ArcFaceModel
from scipy.spatial.distance import euclidean


def load_pickle(path):
    file = open(path,'rb')
    embedding = pickle.load(file)
    name =[]
    for i in embedding.keys():
        name.append(i)
    return name, embedding


class RECOG:
    def __init__(self):
        
        self.align = ALIGN()
        self.embed = "./embds_dict_ad.pkl"
        self.input_size = 112
        self.backbone_type = 'ResNet50'
        self.sub_name = 'arc_res50'
        self.model = ArcFaceModel(size=self.input_size,
                         backbone_type=self.backbone_type,
                         training=False)
        self.ckpt_path = tf.train.latest_checkpoint('/home/cpsd/Documents/Face-Recognition-IBUS_UPDATE_API/checkpoints/' + self.sub_name)


    def recognition(self,image):
     
        if self.ckpt_path is not None:
            print("[*] load ckpt from {}".format(self.ckpt_path))
            self.model.load_weights(self.ckpt_path)
        else:
            print("[*] Cannot find ckpt from {}.".format(self.ckpt_path))
            exit()
        
        name, embedding = load_pickle(self.embed)

        img = image
        bboxes, landmarks, faces = self.align.align_multi(img, min_confidence=0.97, limits=10)
        print("###########",faces.shape)
        crop_face = faces.reshape(112,112,3)
        crop_face = np.array(crop_face,dtype="uint8")
        # try:
        #     print("image_name",names)
        #     cv2.imwrite("./new_crop/"+str(names)+".png",crop_face )
        # except Exception as e:
        #     print("33333333333333333",e)
        bboxes = bboxes.astype(int)
        embs = []
        
        for face in faces:
            if len(face.shape) == 3:
                face = np.expand_dims(face, 0)
            face = face.astype(np.float32) / 255.
            embs.append(l2_norm(self.model(face)).numpy())

        list_min_idx = []
        list_score = []
        for emb in embs:
            dist = [euclidean(emb, embedding[i]) for i in embedding.keys()]
            min_idx = np.argmin(dist)
            list_min_idx.append(min_idx)
            list_score.append(dist[int(min_idx)])
        list_min_idx = np.array(list_min_idx)
       # print("minlist score",list_min_idx)
        list_score = np.array(list_score)
        #print("SSSCCCCOOOORRREEE",list_score)
        
        if list_score[0] < .93:
            list_min_idx[list_score > 1.5] = -1

            return name[list_min_idx[0]]
        else:
            return "Unknown"




from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm
from preprocess import prepare_facebank, load_facebank, align_multi
from scipy.spatial.distance import euclidean
import utils
import time



path =os.getcwd()

file = open(path+"/embds_dict_ad.pkl", 'rb')
embedding = pickle.load(file)
name =[]

for i in embedding.keys():
    name.append(i)

print(name)
def main(frame):
    

    cfg = load_yaml(path+'/configs/arc_res50.yaml')

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

    ckpt_path = tf.train.latest_checkpoint(path+'/checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()
    

    img = frame
    #print(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, landmarks, faces = align_multi(cfg, img, min_confidence=0.97, limits=10)
    ff = faces.reshape(112,112,3)
    #cv2.imwrite("./face.png",ff)
    bboxes = bboxes.astype(int)
    embs = []
    
    for face in faces:
        if len(face.shape) == 3:
            face = np.expand_dims(face, 0)
        face = face.astype(np.float32) / 255.
        embs.append(l2_norm(model(face)).numpy())
        #print("embd",embs)

    list_min_idx = []
    list_score = []
    for emb in embs:
        dist = [euclidean(emb, embedding[i]) for i in embedding.keys()]
        min_idx = np.argmin(dist)
        list_min_idx.append(min_idx)
        list_score.append(dist[int(min_idx)])
    list_min_idx = np.array(list_min_idx)
    print("minlist score",list_min_idx)
    list_score = np.array(list_score)
    print("SSSCCCCOOOORRREEE",list_score)
       
    if list_score[0] < 1:
        list_min_idx[list_score > 1.5] = -1
        print("#############   Face matched   #############")
        for idx, box in enumerate(bboxes):
            print("IDDDDDDDXXXXX",idx)
            frame = utils.draw_box_name(box,
                                        landmarks[idx],
                                        name[list_min_idx[0]],
                                        frame)
    else:
        print("###########     Warning      ##########")
        for idx, box in enumerate(bboxes):
            frame = utils.draw_box_name(box,
                                        landmarks[idx],
                                        "unknown",
                                        frame)
    
    #cv2.imwrite("./target9.png",frame)
    return frame, ff


test_path="/home/intisar/Documents/Arcface_updated/Face-Recognition/test_images/"
files = os.listdir(test_path)
print(files)

for i in range(len(files)):
    #print(i)
    print(files[i])
    image = cv2.imread(test_path+files[i])
    #print(image)
    image, faces =main(image)
    cv2.imwrite("/home/intisar/Documents/Arcface_updated/Face-Recognition/output_images/"+str(i)+".png",image)
    cv2.imwrite("./"+str(i)+".png",faces)

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
a = input("Enter you name:")

def search(a):
    file = open(path+"/embds_dict.pkl", 'rb')
    data = pickle.load(file)
    for i in data.keys():
        if i==a:
            xx=data[i]
            m=a
        else:
            continue
    
    return xx, m

def main():
    

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

    cap = cv2.VideoCapture(0)
   
    target, name=search(a)
    #print('target',target)
    
    while cap.isOpened():

        is_success, frame = cap.read()
        if is_success:
            img = frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes, landmarks, faces = align_multi(cfg, img, min_confidence=0.97, limits=10)
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
                #dist = [euclidean(emb, target) for target in targets]
                dist = [euclidean(emb, target)]
                min_idx = np.argmin(dist)
                list_min_idx.append(min_idx)
                list_score.append(dist[int(min_idx)])
            list_min_idx = np.array(list_min_idx)
            list_score = np.array(list_score)
           # print(list_score)
            if list_score.any()==False:
                continue
            
            # if list_score[0] < 1:
            #     list_min_idx[list_score > FLAGS.threshold] = -1
            #     #print(list_min_idx)
            #     for idx, box in enumerate(bboxes):
            #         frame = utils.draw_box_name(box,
                   
            if list_score[0] < 1:
                list_min_idx[list_score > 1.5] = -1
                print("#############   Face matched   #############")
                for idx, box in enumerate(bboxes):
                    frame = utils.draw_box_name(box,
                                                landmarks[idx],
                                                name,
                                                frame)
            else:
                print("###########     Warning      ##########")
                for idx, box in enumerate(bboxes):
                    frame = utils.draw_box_name(box,
                                                landmarks[idx],
                                                "unknown",
                                                frame)
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow('face Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break    
           
    cap.release()

    cv2.destroyAllWindows()


main()

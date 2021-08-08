import os
import cv2
import time
import utils
import pickle
import datetime
import numpy as np
import tensorflow as tf
from modules.models import ArcFaceModel
from scipy.spatial.distance import euclidean
from modules.utils import set_memory_growth, load_yaml, l2_norm
from preprocess import prepare_facebank, load_facebank, align_multi

path =os.getcwd()

file = open(path+"/embds_dict.pkl", 'rb')
data = pickle.load(file)
names = list(data.keys())

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

    count =0
    skip_frame= 10

    while cap.isOpened():
            print(count)
            if (count% skip_frame == 0):


                strat_time =datetime.datetime.now()
                is_success, frame = cap.read()
                if is_success:
                    img = frame
                bboxes, landmarks, faces = align_multi(cfg, img, min_confidence=0.97, limits=5)
                bboxes = bboxes.astype(int)
                embs = []
                for face in faces:
                    if len(face.shape) == 3:
                        face = np.expand_dims(face, 0)
                    face = face.astype(np.float32) / 255.
                    embs.append(l2_norm(model(face)).numpy())

                list_min_idx = []
                list_score = []
                for emb in embs:
                    dist = [euclidean(emb, target) for target in data.values()]
 
                    min_idx = np.argmin(dist)
                    list_min_idx.append(min_idx)
                    list_score.append(dist[int(min_idx)])
                list_min_idx = np.array(list_min_idx)
                print(list_min_idx)
                list_score = np.array(list_score)

                if list_score.any()==False:
                    continue
                
                if list_score[0] < 1:
                    list_min_idx[list_score > 1.2] = -1
                    print("#############   Face matched   #############\n\n")
                    for idx, box in enumerate(bboxes):
                        
                        print("Name of the Detected Person:\n\n",  names[list_min_idx[idx]]) 
                        print('\n\n')
                        print("***********************************")
                        frame = utils.draw_box_name(box,
                                                    landmarks[idx],
                                                    names[list_min_idx[idx]],
                                                    frame)
                else:
                    print("###########     Warning      ##########")
                    for idx, box in enumerate(bboxes):
                        frame = utils.draw_box_name(box,
                                                    landmarks[idx],
                                                    "unknown",
                                                    frame)

                cv2.imwrite('./save_frame/'+str(strat_time)+'.png',frame)

                cv2.imshow('face Capture', frame)

            count+=1        
          
           
    cap.release()

    cv2.destroyAllWindows()


main()

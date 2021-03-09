#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 22:59:02 2021

@author: sanaullah
"""

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
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
import cv2
import os
import pickle
import numpy as np
from modules.utils import l2_norm
from mtcnn import MTCNN
from align_trans import warp_and_crop_face, get_reference_facial_points
from PIL import Image
from tqdm import tqdm


# flags.DEFINE_string('cfg_path', '/home/sanaullah/Documents/face_recognition-master/configs/arc_res50.yaml', 'config file path')
# flags.DEFINE_string('gpu', '0', 'which gpu to use')
# flags.DEFINE_bool('save', False, 'Whether save')
# flags.DEFINE_float('threshold', 1.5, 'Whether save')
# flags.DEFINE_float('min_confidence', 0.97, 'Whether save')
# flags.DEFINE_bool('update', False, 'whether perform update the facebank')
# flags.DEFINE_string('video', None, 'video file path')


cfg = load_yaml('/home/sanaullah/Documents/face_recognition-master/configs/arc_res50.yaml')
model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

ckpt_path = tf.train.latest_checkpoint('/home/sanaullah/Documents/face_recognition-master/checkpoints/' + cfg['sub_name'])


if ckpt_path is not None:
    print("[*] load ckpt from {}".format(ckpt_path))
    model.load_weights(ckpt_path)
else:
    print("[*] Cannot find ckpt from {}.".format(ckpt_path))
    exit()


def prepare_facebank(cfg, model):
    names = []
    embeddings = []
    detector = MTCNN()
    for name in os.listdir(cfg['face_bank']):
        if os.path.isfile(name):
            continue
        else:
            emb = []
            for file in tqdm(os.listdir(os.path.join(cfg['face_bank'], name))):
                if not os.path.isfile(os.path.join(cfg['face_bank'], name, file)):
                    continue
                else:
                    image = cv2.imread(os.path.join(cfg['face_bank'], name, file))
                   # print(image)
                    image = cv2.resize(image, (cfg['input_size'], cfg['input_size']))
                    face = detector.detect_faces(image)
                    if len(face) > 0:
                        face = face[0]
                        refrence = get_reference_facial_points(default_square=True)

                        landmark = []
                        for _, points in face['keypoints'].items():
                            landmark.append(list(points))

                        warped_face = warp_and_crop_face(image,
                                                         landmark,
                                                         reference_pts=refrence,
                                                         crop_size=(cfg['input_size'], cfg['input_size']))
                        image = np.array(warped_face)

                        image = image.astype(np.float32) / 255.
                        if len(image.shape) == 3:
                            image = np.expand_dims(image, 0)
                        emb.append(l2_norm(model(image)).numpy())
            if len(emb) == 0:
                continue
            emb = np.array(emb)
            mean = np.mean(emb, axis=0)
            embeddings.append(mean)
        names.append(name)
    embeddings = np.array(embeddings)
    names = np.array(names)
    embds_dict = dict(zip(names, embeddings))
    pickle.dump(embds_dict, open("/home/sanaullah/Documents/face_recognition-master/embds_dict.pkl", 'wb'))
    
    # np.save(os.path.join('data', '/home/sanaullah/Documents/face_recognition-master/facebank.npy'), embeddings)
    # np.save(os.path.join('data', '/home/sanaullah/Documents/face_recognition-master/names.npy'), names)

    #return embeddings, names

prepare_facebank(cfg, model)

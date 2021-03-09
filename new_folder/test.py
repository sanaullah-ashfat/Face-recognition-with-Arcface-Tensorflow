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

path="/home/cspd/Documents/face_recognition/"

flags.DEFINE_string('cfg_path', '/home/cspd/Documents/face_recognition/configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_bool('save', False, 'Whether save')
flags.DEFINE_float('threshold', 1.5, 'Whether save')
flags.DEFINE_float('min_confidence', 0.97, 'Whether save')
flags.DEFINE_bool('update', False, 'whether perform update the facebank')
flags.DEFINE_string('video', None, 'video file path')


def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

    ckpt_path = tf.train.latest_checkpoint('/home/cspd/Documents/face_recognition/checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()

    if FLAGS.update:
        print('Face bank updating...')
        targets, names = prepare_facebank(cfg, model)
        print('Face bank updated')
    else:
        targets, names = load_facebank(path)
        print('Face bank loaded')

    if FLAGS.video is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(str(FLAGS.video))

    while cap.isOpened():

        is_success, frame = cap.read()
        if is_success:
            img = frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes, landmarks, faces = align_multi(cfg, img, min_confidence=FLAGS.min_confidence, limits=10)
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
                dist = [euclidean(emb, target) for target in targets]
                min_idx = np.argmin(dist)
                list_min_idx.append(min_idx)
                list_score.append(dist[int(min_idx)])
            list_min_idx = np.array(list_min_idx)
            list_score = np.array(list_score)
            #print(list_score)
            if not list_score:
                continue
            
            if list_score[0] < 1:
                list_min_idx[list_score > FLAGS.threshold] = -1
                #print(list_min_idx)
                for idx, box in enumerate(bboxes):
                    frame = utils.draw_box_name(box,
                                                landmarks[idx],
                                                names[list_min_idx[idx] + 1],
                                                frame)
            else:
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

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

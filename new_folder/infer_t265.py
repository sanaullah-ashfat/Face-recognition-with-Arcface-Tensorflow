from absl import app, flags, logging
import pyrealsense2 as rs
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

flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_bool('save', False, 'Whether save')
flags.DEFINE_float('threshold', 1.5, 'Whether save')
flags.DEFINE_float('min_confidence', 0.97, 'Whether save')
flags.DEFINE_bool('update', False, 'whether perform update the facebank')
flags.DEFINE_string('video', None, 'video file path')


def main(_argv):
    pipe = rs.pipeline()

    # Build config object and request pose data
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color)

    # Start streaming with requested config
    pipe.start(cfg)

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

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
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
        targets, names = load_facebank(cfg)
        print('Face bank loaded')

    if FLAGS.save:
        video_writer = cv2.VideoWriter('./recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 480))
        # frame rate 6 due to my laptop is quite slow...

    while True:

        frame = pipe.wait_for_frames()

        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes, landmarks, faces = align_multi(cfg, img, min_confidence=FLAGS.min_confidence, limits=3)
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
        list_min_idx[list_score > FLAGS.threshold] = -1
        for idx, box in enumerate(bboxes):
            frame = utils.draw_box_name(box,
                                        landmarks[idx],
                                        names[list_min_idx[idx] + 1],
                                        frame)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('face Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if FLAGS.save:
            video_writer.write(frame)

        if key == ord('q'):
            break

    if FLAGS.save:
        video_writer.release()
    pipe.stop()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

import cv2
import os
import numpy as np

from modules.utils import l2_norm
from mtcnn import MTCNN
from align_trans import warp_and_crop_face, get_reference_facial_points
from PIL import Image
from tqdm import tqdm


def prepare_facebank(cfg, model):
    names = ['Unknown']
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

    np.save(os.path.join('data', 'facebank.npy'), embeddings)
    np.save(os.path.join('data', 'names.npy'), names)

    return embeddings, names


def load_facebank(path):
    embeddings = np.load(os.path.join(path+ 'facebank.npy'))
    names = np.load(os.path.join(path+'names.npy'))
    return embeddings, names


def align_multi(cfg, image, min_confidence=0.99, limits=None):
    boxes = []
    landmarks = []
    detector = MTCNN()
    faces = detector.detect_faces(image)
    refrence = get_reference_facial_points(default_square=True)
    for face in faces:
        if face['confidence'] < min_confidence:
            continue
        boxes.append(face['box'])
        landmark = []
        for name, points in face['keypoints'].items():
            landmark.append(list(points))
        landmarks.append(landmark)
    if limits:
        boxes = boxes[:limits]
        landmarks = landmarks[:limits]

    faces = []
    for landmark in landmarks:
        warped_face = warp_and_crop_face(image,
                                         landmark,
                                         reference_pts=refrence,
                                         crop_size=(cfg['input_size'], cfg['input_size']))
        faces.append(warped_face)

    return np.array(boxes), np.array(landmarks), np.array(faces)


def align(cfg, image):
    detector = MTCNN()
    face = detector.detect_faces(image)[0]
    refrence = get_reference_facial_points(default_square=True)

    landmark = []
    for name, points in face['keypoints'].items():
        landmark.append(list(points))

    warped_face = warp_and_crop_face(image,
                                     landmark,
                                     reference_pts=refrence,
                                     crop_size=(cfg['input_size'], cfg['input_size']))
    return warped_face

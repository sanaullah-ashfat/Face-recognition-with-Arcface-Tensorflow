import cv2
import os
import numpy as np
from src.align_trans import warp_and_crop_face, get_reference_facial_points
from src.align.test import DETECTION
from PIL import Image
from tqdm import tqdm


class ALIGN:

    def __init__ (self):
        
        self.detector = DETECTION()


    def align_multi(self, image, min_confidence=0.6, limits=None):
        boxes = []
        landmarks = []
        print(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box, land = self.detector.detect(image)
        refrence = get_reference_facial_points(default_square=True)
        for face in box:
            if face[-1] < min_confidence:
                continue
            print(face[:4])
            
            boxes.append(face[:4])
            landmark = []
            
            for points in land:
                landmark.append(list(points))
            landmarks.append(landmark)
        if limits:
            boxes = boxes[:limits]
            landmarks = landmarks[:limits]
            

        faces = []
        for landmark in landmarks:
            print("************",landmark)
            warped_face = warp_and_crop_face(image,
                                            landmark,
                                            reference_pts=refrence,
                                            crop_size=(112, 112))
            faces.append(warped_face)


        return np.array(boxes), np.array(landmarks), np.array(faces)
    
    def align(self, image):
       
        face = self.detector.detect_faces(image)[0]
        refrence = get_reference_facial_points(default_square=True)

        landmark = []
        for name, points in face['keypoints'].items():
            landmark.append(list(points))

        warped_face = warp_and_crop_face(image,
                                        landmark,
                                        reference_pts=refrence,
                                        crop_size=(112, 112))
        return warped_face





import cv2
import glob
import base64
import numpy as np
import pickle
from main.RECOGNITION import RECOG
from main.embed_save import SAVE_EMBDED


class TASK:
    def __init__(self):
        self.emb =SAVE_EMBDED()
        self.recg=RECOG()

    def response(self,image,task=None,ID= None):
        if task == "recognition":

            result=self.recg.recognition(image)
            return result

        if task == "register": 

            return self.emb.single_image(image, ID)


import os
import cv2
import glob
import time
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from src.align.test import DETECTION
import tensorflow as tf
from src.modules import utils
from src.ALIGNMENT import ALIGN
from src.modules.utils import l2_norm
from scipy.spatial.distance import euclidean
from src.modules.models import ArcFaceModel
from src.modules.evaluations import get_val_data, perform_val
from src.modules.utils import set_memory_growth, load_yaml, l2_norm
from src.align_trans import warp_and_crop_face, get_reference_facial_points



class SAVE_EMBDED:
    def __init__(self):
        
        self.align = ALIGN()
        self.detector = DETECTION()
        self.embed = "./embds_dict_ad.pkl"
        self.input_size = 112
        self.backbone_type = 'ResNet50'
        self.sub_name = 'arc_res50'
        self.model = ArcFaceModel(size=self.input_size,
                         backbone_type=self.backbone_type,
                         training=False)
        self.ckpt_path = tf.train.latest_checkpoint('/home/cpsd/Documents/Face-Recognition-IBUS_UPDATE_API/checkpoints/' + self.sub_name)


    def save_multiple_embed(self,path):
        white=[255,255,255]
     
        
        if self.ckpt_path is not None:
            print("[*] load ckpt from {}".format(self.ckpt_path))
            self.model.load_weights(self.ckpt_path)
        else:
            print("[*] Cannot find ckpt from {}.".format(self.ckpt_path))
            exit()
        
        names = []
        emb=[]
        embeddings=[]
        for image_name in glob.glob(path+"*"):
            print(image_name)
            if not image_name:
                continue
            else:
                name = image_name.split("/")[-1].split(".")[0]
                image = cv2.imread(image_name)
                
                #print(image)
                #image= cv2.copyMakeBorder(image,150,150,150,150,cv2.BORDER_CONSTANT,value=white)
                box, land, face =self.align.align_multi(image, min_confidence=0.97, limits=10)
                # img = np.array(face,dtype="uint8")
                # cv2.imwrite("./crop/"+str(name)+".png",img )
                # print(face.shape)
                image = face.astype(np.float32) / 255.
                #print(image.shape)
                # mirror = face.reshape(112,112,3)
                # mirror= cv2.flip(mirror, 1)
                # mirror= mirror.astype(np.float32) / 255.
                # mirror= mirror.reshape(1,112,112,3)
                # print(mirror.shape)
                if len(image.shape) == 3:
                    image = np.expand_dims(image, 0)
                    #mirror= np.expand_dims(mirror, 0)
               
                emb.append(l2_norm(self.model(image)).numpy()[0])
                names.append(name)
              
            
                
        # print("number of emb",len(emb))
        embd = np.asarray(emb)
        # print(" arry number of embd",embd.shape)

        nam = np.array(names)
        # print("*********** namea ************:",nam)
        embds_dict = dict(zip(nam, embd))
        # print("*********** embds_dict ************:",embds_dict)
        print("*************    Prepareing embedding for save     **************")
        print("*************     Embedding save Successfully    **************")
        with open("./embds_dict_ad.pkl", "wb") as fi:
            bin_obj = pickle.dumps(embds_dict)
            fi.write(bin_obj)

        #return pickle.dump((embds_dict), open("./embds_dict_ad.pkl", 'ab'))
        return embds_dict

        #return pickle.dump(embds_dict, open("./embds_dict_ad.pkl", 'wb'))


    def single_image(self,image, name):
        white=[255,255,255]
        emb=[]
        names=[]
        
        if self.ckpt_path is not None:
            print("[*] load ckpt from {}".format(self.ckpt_path))
            self.model.load_weights(self.ckpt_path)
        else:
            print("[*] Cannot find ckpt from {}.".format(self.ckpt_path))
            exit()

        #image= cv2.copyMakeBorder(image,150,150,150,150,cv2.BORDER_CONSTANT,value=white)
        box, land, face =self.align.align_multi(image, min_confidence=0.97, limits=10)

        image = face.astype(np.float32) / 255.
        #print(image.shape)
        # mirror = face.reshape(112,112,3)
        # mirror= cv2.flip(mirror, 1)
        # mirror= mirror.astype(np.float32) / 255.
        # mirror= mirror.reshape(1,112,112,3)
        # print(mirror.shape)
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
            #mirror= np.expand_dims(mirror, 0)
        
        emb.append(l2_norm(self.model(image)).numpy()[0])
        names.append(name)
           
        # print("number of emb",len(emb))
        embd = np.asarray(emb)
        # print(" arry number of embd",embd.shape)

        nam = np.array(names)
        # print("*********** namea ************:",nam)
        embds_dict = dict(zip(nam, embd))
        # print("*********** embds_dict ************:",embds_dict)
        print("*************    Prepareing embedding for save     **************")
        print("*************     Embedding save Successfully    **************")
        with open("./embds_dict_ad.pkl", "wb") as fi:
            bin_obj = pickle.dumps(embds_dict)
            fi.write(bin_obj)

        #return pickle.dump((embds_dict), open("./embds_dict_ad.pkl", 'ab'))
        return embds_dict


        
        


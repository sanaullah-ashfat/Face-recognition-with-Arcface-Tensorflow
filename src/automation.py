

import cv2
import sys
import os
import traceback
from PIL import Image
from glob import glob
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import numpy as np
#%matplotlib inline
import matplotlib.image as mpimg

CASCADE="/home/cspd/Documents/IBUS/Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)




def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

# def transform_image(img,ang_range,shear_range,trans_range,path,image_name):

#     # Rotation

#     # ang_rot = np.random.uniform(ang_range)-ang_range/2
#     rows,cols,ch = img.shape    
#     # Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

#     # Translation
#    # tr_x = trans_range*np.random.uniform()-trans_range/2
#     #tr_y = trans_range*np.random.uniform()-trans_range/2
#     #Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

#     # Shear
#     pts1 = np.float32([[5,5],[20,5],[5,20]])

#     pt1 = 5+shear_range*np.random.uniform()-shear_range/2
#     pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
#     # Brightness 
    

#     pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

#     shear_M = cv2.getAffineTransform(pts1,pts2)
        
#    # img = cv2.warpAffine(img,Rot_M,(cols,rows))
#     #img = cv2.warpAffine(img,Tran8s_M,(cols,rows))
#     img = cv2.warpAffine(img,shear_M,(cols,rows))
    
#     img = augment_brightness_camera_images(img)
    
#     cv2.imwrite(os.path.join(path,image_name+"br"+".jpg"), img)
    
#    # return img


def transform_image(img,shear_range,path,image_name):

    rows,cols,ch = img.shape    

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
    # Brightness 
    
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)        
   # img = cv2.warpAffine(img,Rot_M,(cols,rows))
    #img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    #img = augment_brightness_camera_images(img)
    
    cv2.imwrite(os.path.join(path,image_name+"sher"+".png"), img)
    
   # return img



def rotate_blar(img,path,image_name):
    
    for i in range(-45,45,10):
        image =rotate(img,i)
        image=cv2.blur(image,(5,5))
        image=cv2.resize(image,(112,112))
        cv2.imwrite(os.path.join(path,image_name+"blur_"+str(i)+".png"), image)




def rotate_image(img,path,image_name):
    
    for i in range(-45,45,10):
        image =rotate(img,i)
        image=cv2.resize(image,(112,112))
        cv2.imwrite(os.path.join(path,image_name+str(i)+".png"), image)




def increase_bright(img,path,image_name):
    
    for i in range(-45,45,10):
        image =rotate(img,i)
        image=cv2.resize(image,(112,112))
        image= cv2.detailEnhance(image, sigma_s=2, sigma_r=0.7)
        
        cv2.imwrite(os.path.join(path,image_name+"bright_"+str(i)+".png"), image)


def increase_contrast(img,path,image_name):
    
    for i in range(-45,45,10):
        image =rotate(img,i)
        lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
        cl = clahe.apply(l)
        image = cv2.merge((cl,a,b))
        image=cv2.resize(image,(112,112))
        image= cv2.detailEnhance(image, sigma_s=2, sigma_r=0.7)
        cv2.imwrite(os.path.join(path,image_name+"con_"+str(i)+".png"), image)



def augment(img, img_name):
    
    img =cv2.imread(img)
    x=create_directory(img_name)
    rotate_image(img,x,img_name)
    #increase_contrast(img,x,img_name)
    rotate_blar(img,x,img_name)
    increase_bright(img,x,img_name)
    
# Create directory
def create_directory(folder_name):
    dirName = "/home/cspd/Documents/IBUS/process_data/"+folder_name
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

    return dirName+"/"


def detect_faces(image_path):
    
    image=cv2.imread(image_path)
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
    image_name = image_path.split('/')[-1].split('.')[-2]

    for x,y,w,h in faces:
        sub_img = image[y-10:y+h+10,x-10:x+w+10]
        sub_img = cv2.resize(sub_img, (112,112))
        cv2.imwrite("/home/cspd/Documents/IBUS/tmp/" + image_name+".png",sub_img)




def file_read(input_path):
    return glob(input_path)




def main(image_path, tmp_path):
    l = file_read(image_path)
    for file in l:
        detect_faces(file)
    m = file_read(tmp_path)
    for file in m:
        image_name = file.split('/')[-1].split('.')[-2]
        augment(file, image_name)
        
        
        
        

if __name__ == '__main__':
    main("/home/cspd/Documents/IBUS/new/*.png", "/home/cspd/Documents/IBUS/tmp/*.png")






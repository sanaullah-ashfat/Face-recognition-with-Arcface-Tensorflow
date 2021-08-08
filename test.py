import cv2
import glob
import pickle
from main.RECOGNITION import RECOG
from main.embed_save import SAVE_EMBDED

emb =SAVE_EMBDED()
recg=RECOG()
# img= cv2.imread("./abc.jpg")
with open ("././embds_dict_ad.pkl","rb") as file:
    lo = pickle.load(file)
print(len(lo))
# white = [240,240,240]

# path ="/home/cpsd/Downloads/All/images_nid/select/"
path ="/home/cpsd/Downloads/Images/"
# # # train =emb.prepare_facebank(path)
# names=[]
# for image_name in glob.glob(path+"*"):
#     name = image_name.split("/")[-1].split(".")[0]
#     try:
#         image= cv2.imread(image_name)
#         #image= cv2.copyMakeBorder(image,150,150,150,150,cv2.BORDER_CONSTANT,value=white)
#         print(image_name)
#         name = image_name.split("/")[-1].split(".")[0]
#         cv2.imwrite("./padding/"+str(name)+".jpg",image)
#         result=recg.recognition(image,name)
#         names.append(name)
#     except:
#         print(name)
# # print(len(names))
# # # # names=[]
# # # # from PIL import Image
# # # # for image_name in glob.glob(path+"*"):
# # # #     # image= cv2.imread(image_name)
# # # #     print(image_name)
# # # #     name = image_name.split("/")[-1].split(".")[0]
# # # #     names.append(name)

# # # #     im = Image.open(image_name)
# # # #     rgb_im = im.convert('RGB')
# # # #     rgb_im.save("./"+str(name)+",png")

# # # # print(len(names))
# # # # img= cv2.imread("/home/cpsd/Downloads/Images/ha.jpg")
# # # # result=recg.recognition(img,None)
# # # # print(result)

#train =emb.save_multiple_embed(path)


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


def extract_data(self,path,out_path):

    names = []
    emb=[]
    embeddings=[]
    img_list = glob.glob(path+"/*")
    print(img_list)

    for i in range(len(img_list)):
        image = img_list[i]
        image_name= image.split(".")[-1]
        img_name = image.split("/")[-1].split(".")[0]
        print(img_name)
        image = cv2.imread(image)
        image = cv2.resize(image,(512,512))
        try:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            face, bbox,_ = self.detector.get_cropped_face(image)
            print(len(face))
            image= face[0]
            # print(bbox,bbox)
            cv2.imwrite("./image.jpg",image)
            # cv2.imwrite("./images.jpg",face[1])
            k, idx = self.recognition(image,bbox)
            print("*********************************************",idx)
            print("*********************************************",k)
            if k == "Known":
                if not os.path.isdir("/home/sanaullah/Documents/facebank/micro_service_update/folder/"+str(idx)):
                    os.mkdir("/home/sanaullah/Documents/facebank/micro_service_update/folder/"+str(idx))
                    # print("---------------------------------------------",path_out)
                    path_out ="/home/sanaullah/Documents/facebank/micro_service_update/folder/"+str(idx)+"/"
                    cv2.imwrite(path_out+str(img_name)+".png",image )#cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                    #return "WOW The system work well"
                else:
                    path_out ="/home/sanaullah/Documents/facebank/micro_service_update/folder/"+str(idx)+"/"
                    cv2.imwrite(path_out+str(img_name)+".png",image )
                    #return "WOW The system work well"
            else:

                images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = images.astype(np.float32) / 255.
                #mirror = image.reshape(112,112,3)
                mirror= cv2.flip(image, 1)
                mirror= mirror.astype(np.float32) / 255.
                if len(image.shape) == 3:
                    image = np.expand_dims(image, 0)
                    mirror= np.expand_dims(mirror, 0)
                image = l2_norm(self.model(image))
                mirror = l2_norm(self.model(mirror))
                ebd = np.mean([image, mirror],axis=0)
                emb.append(ebd)
                names.append(img_name)
                embd = np.asarray(emb)
                nam = np.array(names)
                embds_dict = dict(zip(nam, embd))
                with open("./project/information/embdding.pkl", "wb") as fi:
                    bin_obj = pickle.dumps(embds_dict)
                    fi.write(bin_obj)
                if not os.path.isdir("/home/sanaullah/Documents/facebank/micro_service_update/folder/"+str(img_name)):
                    os.mkdir("/home/sanaullah/Documents/facebank/micro_service_update/folder/"+str(img_name))
                    path_out = "/home/sanaullah/Documents/facebank/micro_service_update/folder/"+str(img_name)+"/"
                    print("######################################",img_name,path_out)
                    cv2.imwrite(path_out+str(img_name)+".png", images)
                else:
                    path_out = "/home/sanaullah/Documents/facebank/micro_service_update/folder/"+str(img_name)+"/"
                    print("######################################",img_name,path_out)
                    cv2.imwrite(path_out+str(img_name)+".png", images)

                #return "sucessfull"
        except Exception as e:
            print(e)


import pickle
file = open("embds_dict_ad.pkl",'rb')
object_file = pickle.load(file)
print(object_file.keys())
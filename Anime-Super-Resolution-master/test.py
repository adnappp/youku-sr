import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam
import random
import os
from model import wdsr_a, wdsr_b
from utils import DataLoader
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
model = wdsr_b(scale=4, num_res_blocks=32,num_filters=64)
model.load_weights('./weights/weights_all_64/wdsr-b-32-x4.h5')
#model.load_weights('./train_weights/wdsr-b-32-x_all_47.h5')

data_loader = DataLoader(scale=4)
def test(model, setpath='../test', name='evaluate'):
    out_path = "../result_all_64"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    floders = os.listdir(setpath)
    floders.sort()
    all_floder = floders[:5]
    sub_floder = floders[5:]
    for all_f in all_floder:
        name = all_f[:12]+"h_Res"
        #f2 = ../test/Youku_00200_l/
        f2 = os.path.join(setpath,all_f)
        images = os.listdir(f2)
        #savepath=../result/Youku_00200_h_Res
        save_path=os.path.join(out_path,name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for image in images:
            print(os.path.join(f2,image))
            lr = data_loader.imread(os.path.join(f2,image))
            lr = np.asarray(lr)
            sr = model.predict(np.array([lr]))[0]
            sr = np.clip(sr, 0, 255)
            sr = sr.astype('uint8')
            sr = Image.fromarray(sr)
            sr.save(os.path.join(save_path,image))
    for sub_f in sub_floder:
        name = sub_f[:12]+"h_Sub25_Res"
        #f2 = ../test/Youku_00200_l/
        f2 = os.path.join(setpath,sub_f)
        images = os.listdir(f2)
        #savepath=../result/Youku_00200_h_Res
        save_path=os.path.join(out_path,name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        i=0
        for image in images:
            #if i%25==0:
            print(os.path.join(f2, image))
            lr = data_loader.imread(os.path.join(f2,image))
            lr = np.asarray(lr)
            sr = model.predict(np.array([lr]))[0]
            sr = np.clip(sr, 0, 255)
            sr = sr.astype('uint8')
            sr = Image.fromarray(sr)
            sr.save(os.path.join(save_path,image))
            i+=1
    pass
test(model)

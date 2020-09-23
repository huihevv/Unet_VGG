import numpy as np
from model import vgg19, unet

import os
from PIL import Image
import argparse
from keras import backend as K

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import numpy as np
from PIL import Image
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from unetdata import *
import keras
from drop_block import DropBlock2D
from keras.layers.core import Activation
from keras.callbacks import TensorBoard
#from keras.utils import plot_model
#from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
import time
def jd_loss(y_true, y_pred):
    y_true = keras.layers.Reshape([-1])(y_true)
    y_pred = keras.layers.Reshape([-1])(y_pred)
    tp = keras.layers.Multiply()([y_true, y_pred])
    tp = K.sum(tp, axis=1)
    t = keras.layers.Multiply()([y_true, y_true])
    t = K.sum(t, axis=1)
    p = keras.layers.Multiply()([y_pred, y_pred])
    p = K.sum(p, axis=1)
    temp = keras.layers.Subtract()([(keras.layers.Add()([t, p])), tp])
    # temp = keras.layers.Add()([temp, 1e-9])
    # temp = 1/temp
    # temp = keras.layers.Multiply()([tp,temp])
    jd_l = 1 - (tp / (temp + 1e-9))
    return jd_l


class myUnet(object):
    def __init__(self, img_rows=128, img_cols=128):
        self.img_rows = img_rows
        self.img_cols = img_cols
    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def train(self):
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print('******')
        print(np.shape(imgs_mask_train))
        # vgg = vgg19()
        # model = unet(vgg)
        filelist = glob.glob('./data_new/model/*.hdf5')
        num = len(filelist)
        # midname = filelist[num - 1][filelist[num - 1].rindex("\\") + 1:]
        midname = filelist[num - 1]
        # model = load_model('./data_new/model/' + midname)
        model = load_model(midname, custom_objects={'jd_loss':jd_loss, 'DropBlock2D': DropBlock2D})
        print('load model is')
        print(midname)
        print("got unet")

        print('predict test data')
        time_teststart = time.time()
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        time_testover = time.time()
        print('test time: ', time_testover - time_teststart)
        np.save("data_new/results/data_self/imgs_mask_test.npy", imgs_mask_test)


    def save_img(self):
        print("array to image")
        imgs = np.load('./data_new/results/data_self/imgs_mask_test.npy')
        #print(imgs)
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("./data_new/results/jpg_self/%d.jpg" % (i))


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()


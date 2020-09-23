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
from keras.layers.core import Activation
from keras.callbacks import TensorBoard
#from keras.utils import plot_model
#from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
import time

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
        vgg = vgg19()
        model = unet(vgg)

        vgg.summary()
        model.summary()

        #model = load_model('./data_new/model/weights.19-0.635726.hdf5')
        #model_checkpoint = ModelCheckpoint('data_new/model/unetvgg.hdf5', monitor='loss', verbose=1, save_best_only=True)
        model_checkpoint = ModelCheckpoint('./data_new/model/weights.{epoch:02d}-{val_loss:.6f}.hdf5', monitor='loss',
                                           verbose=1, save_best_only=True, period=1)
        print('Fitting model...')
        # model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=1, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, TensorBoard(log_dir='./tmp/log')])

        tbCallBack = keras.callbacks.TensorBoard(log_dir='./tmp/log', histogram_freq=0, write_graph=True,
                                                 write_images=True)
        time_start = time.time()
        model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=100, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, tbCallBack])
        time_over = time.time()
        print('training time :', time_over - time_start)
        print('predict test data')
        time_startnew = time.time()
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        time_overnew = time.time()
        print('test time :', time_overnew - time_startnew)
        np.save("data_new/results/data_self/imgs_mask_test.npy", imgs_mask_test)


    def save_img(self):
        print("array to image")
        imgs = np.load('./data_new/results/data_self/imgs_mask_test.npy')
        print(imgs)
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("./data_new/results/jpg_self/%d.jpg" % (i))


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()


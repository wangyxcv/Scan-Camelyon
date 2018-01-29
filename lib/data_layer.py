import _init_path
import caffe
import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy.misc import imresize
from camelyon_dataset import CamelyonDataset
import os
import random
class CamelyonDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(top)!=2:
            raise Exception("You should define data and label")

        if len(bottom)!=0:
            raise Exception("We do not need bottom")

        params = eval(self.param_str)
        
        self.batch_size = params["batch_size"]
        self.im_shape = params["im_shape"]
        
        no_count = ["patient_017_node_4", "patient_051_node_2", "patient_046_node_4", "patient_064_node_0", "patient_096_node_0"]
        
        tif_root = params["tif_root"]
        tif_files = []
        for x in sorted(os.listdir(tif_root)):
            if x.split(".")[0] not in no_count:
                tif_files.append(os.path.join(tif_root, x))

        xml_root = params["xml_root"]
        xml_files = []
        for x in sorted(os.listdir(xml_root)):
            if x.split(".")[0] not in no_count:
                xml_files.append(os.path.join(xml_root, x))

        self.sampler = CamelyonDataset(tif_files, xml_files, 244, 0.1) #345

        
        top[0].reshape(self.batch_size, 3, self.im_shape, self.im_shape)
        top[1].reshape(self.batch_size)

        
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        data = np.zeros((self.batch_size, self.im_shape, self.im_shape, 3))
        label = np.zeros((self.batch_size))
        for i in range(self.batch_size):
            image_patch, mask = self.sampler.next()

            #rand_rotate = random.randint(1, 360)
            #image_patch = image_rotate(image_patch, rand_rotate, self.im_shape)
            #mask = image_rotate(mask, rand_rotate, self.im_shape)
            
            #rand_flip = random.randint(-1,2)
            #if rand_flip is not 2:
                #image_patch = cv2.flip(image_patch, rand_flip)
                #mask = cv2.flip(mask, rand_flip)
            
            data[i] = image_patch.copy()
            mask = mask[106:138, 106:138]
            if np.bincount(mask.flatten())[0] < 32*32*0.5:
                label[i] = 1
            #cv2.imshow("image_patch", image_patch)
            #cv2.imshow("mask", mask)
            #cv2.waitKey(2000)
            
        data = data.transpose((0, 3, 1, 2))

        top[0].data[...] = data.copy()
        top[1].data[...] = label.copy()


    def backward(self, top, propagate_down, bottom):
        pass
def image_rotate(img, angle, shape):
    rotate_matrix = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2), angle, 1)
    img_rotate = cv2.warpAffine(img, rotate_matrix, (img.shape[0], img.shape[1]))
    centre = img.shape[0]/2
    img_rotate = img_rotate[centre-shape/2:centre+shape/2, centre-shape/2:centre+shape/2]
    return img_rotate

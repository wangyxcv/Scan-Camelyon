import numpy as np
import time
import multiprocessing
import scipy.io as sio
import signal
import argparse
import sys
sys.path.append("/opt/intern/users/yuewang/ScanNet-FCN/caffe/python/")
sys.path.append("/usr/local/bin/")
import caffe

import multiresolutionimageinterface as mir
from matplotlib import pyplot as plt
import cv2
import os
import math
import random
import pdb
from camelyon_dataset import CamelyonDataset

class CaffeNet:
    def __init__(self, model_def, model_weights, device_id, input_size=None):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)

        self.net = caffe.Net(model_def,
                    model_weights,
                    caffe.TEST)

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        #mu = np.array([104.0, 117.0, 123.0], dtype=np.float32)
        self.transformer.set_transpose('data', (2,0,1))
        #self.transformer.set_mean('data', mu)
        #self.transformer.set_raw_scale('data', 255)
        #self.transformer.set_channel_swap('data', (2,1,0))

    def get_feature(self, image):
        #image = caffe.io.load_image(filename)
        transformed_image = self.transformer.preprocess('data', image)
        #pdb.set_trace()
        self.net.blobs['data'].data[...] = transformed_image
        output = self.net.forward()
        return output["softmax"][0][1]

def find_boxes(position):
    boxes = []
    begin = 0
    for i in range(1,len(position[0])):
        
        if abs(position[0][i]-position[0][i-1]) > 10 or i == len(position[0])-1:
            box = [0,0,0,0]
            box[1] = np.min(position[0][begin:i-1])
            box[0] = np.min(position[1][begin:i-1])
            box[3] = np.max(position[0][begin:i-1])
            box[2] = np.max(position[1][begin:i-1])
            boxes.append(box)
            begin = i
        
    return boxes
def find_boxes_image(image):
    boxes = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 0:
                box = [0,0,0,0]
                box[1] = i
                box[0] = j
                for ii in range(i, image.shape[0]):
                    if image[ii][j] > 0:
                        box[3] = ii
                for jj in range(j, image.shape[1]):
                    if image[i][jj] > 0:
                        box[2] = jj
                image[i:ii, j:jj] = 255
                boxes.append(box)
    return boxes

test_net = "/opt/intern/users/yuewang/ScanNet-FCN/model/test_py.prototxt"
#weights = "/opt/intern/users/yuewang/ScanNet-FCN/output/model/ScanNet_iter_40000.caffemodel"
weights = "/opt/intern/users/yuewang/ScanNet-FCN/output/model/ScanNet_random_list_iter_130000.caffemodel"
gpu_id = 6
net = CaffeNet(test_net, weights, gpu_id)
path = "/opt/intern/users/yuewang/dataset/Camelyon17/"
r = mir.MultiResolutionImageReader()
#files = os.listdir(path+"gt")
files = ["patient_017_node_4", "patient_039_node_1", "patient_046_node_4", "patient_064_node_0", "patient_089_node_3"]
tif_files = [os.path.join(path, "images", x+".tif") for x in files]
xml_files = [os.path.join(path, "label", x+".xml") for x in files]
sampler = CamelyonDataset(tif_files, xml_files, 1204, 0.5)
while True:
    image_patch, gt = sampler.next()
    output = net.get_feature(image_patch)
    plt.subplot(1,3,1)
    plt.imshow(output, cmap="gray")
    plt.subplot(1,3,2)
    plt.imshow(gt[106:1204-106, 106:1204-106], cmap="gray")
    plt.subplot(1,3,3)
    plt.imshow(image_patch)
    plt.show()

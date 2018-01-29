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
from camelyon_test import MRImage


class CaffeNet:
    def __init__(self, model_def, model_weights, device_id, input_size=None):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)

        self.net = caffe.Net(model_def,
                    model_weights,
                    caffe.TEST)

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        self.transformer.set_transpose('data', (2,0,1))

    def get_feature(self, image):
        transformed_image = self.transformer.preprocess('data', image)
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
weights = "/opt/intern/users/yuewang/ScanNet-FCN/output/model/ScanNet_random_list_iter_200000.caffemodel"
gpu_id = 7
net = CaffeNet(test_net, weights, gpu_id)
path = "/data/yuewang/CAMELYON17/testing/centres/"

files = os.listdir(path)
stride = 32*31
size = 1204
thr = 0.5
tt = time.time()
for file in files:
    img_id = int(file.split("_")[1])-100
    if img_id/20 != 4:
        continue
    t = time.time()
    print "processing:", file
    test_count = 0
    tif_file = os.path.join(path, file)

    sampler = MRImage(tif_file)
    
    shape = sampler.max_size
    print "image shape:", shape
    stride_h = int(math.ceil((shape[0]-212)/stride))
    stride_w = int(math.ceil((shape[1]-212)/stride))
    
    out_shape = (int(shape[1]/32)+31, int(shape[0]/32)+31)
    mask_map = np.zeros(out_shape)
    out_map = np.zeros(out_shape)
    
    for i in range(stride_h):
        for j in range(stride_w):
            x1 = i*stride
            y1 = j*stride  
            image_patch = sampler.sub_region((x1, y1, x1+size-1, y1+size-1))
            if image_patch is None:
                continue
            output = net.get_feature(image_patch)
            test_count += 1

            out_map[j*31:(j+1)*31, i*31:(i+1)*31] = output
    
    print "cost time:", (time.time()-t)/60, "min"
    print "test", test_count, "image_patch"
    cv2.imwrite("output/prediction/random_list/test/"+file.split(".")[0]+".jpg", out_map*200)

print "all cost time:", (time.time()-tt)/3600, "h"

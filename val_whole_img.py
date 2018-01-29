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
from camelyon_dataset import MRImageSampler


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
weights = "/opt/intern/users/yuewang/ScanNet-FCN/output/model/ScanNet_sample_image_rotate_iter_50000.caffemodel"
gpu_id = 7
net = CaffeNet(test_net, weights, gpu_id)
path = "/opt/intern/users/yuewang/dataset/Camelyon17/"

files = ["patient_051_node_2", "patient_017_node_4", "patient_039_node_1", "patient_046_node_4", "patient_064_node_0", "patient_089_node_3"]

#files = ["patient_046_node_4"]
#files = os.listdir(path+"images")
stride = 32*31
size = 1204
thr = 0.5

for file in files:
    t = time.time()
    print file
    #file = file.split(".")[0]
    test_count = 0
    tif_file = os.path.join(path, "images", file+".tif")
    xml_file = os.path.join(path, "label", file+".xml")

    sampler = MRImageSampler(tif_file, xml_file, size)
    
    shape = sampler.max_size
    stride_h = int(math.ceil((shape[0]-212)/stride))
    stride_w = int(math.ceil((shape[1]-212)/stride))
    
    out_shape = (int(shape[1]/32)+31, int(shape[0]/32)+31)
    mask_map = np.zeros(out_shape)
    out_map = np.zeros(out_shape)
    
    for i in range(stride_h):
        for j in range(stride_w):
            x1 = i*stride
            y1 = j*stride  
            result = sampler.sub_region((x1, y1, x1+size-1, y1+size-1))
            if result is None:
                continue
            image_patch = result[0]
            mask = result[1][106:size-106, 106:size-106]
            output = net.get_feature(image_patch)
            test_count += 1

            mask = cv2.resize(mask, (31,31), interpolation = cv2.INTER_LINEAR)
            mask_map[j*31:(j+1)*31, i*31:(i+1)*31] = mask
            out_map[j*31:(j+1)*31, i*31:(i+1)*31] = output
            '''
            plt.subplot(2,2,1)
            plt.imshow(mask, cmap='gray')
            plt.subplot(2,2,2)
            plt.imshow(output, cmap='gray')
            plt.subplot(2,2,3)
            plt.imshow(mask_map, cmap='gray')
            plt.subplot(2,2,4)
            plt.imshow(out_map, cmap='gray')
            plt.show()
            '''
    print "time:", time.time()-t
    '''
    plt.subplot(1,2,1)
    plt.imshow(mask_map, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(out_map, cmap='gray')
    plt.show()
    '''
    print test_count
    cv2.imwrite("output/prediction/out_rotate_"+file+".jpg", out_map*200)
    #cv2.imwrite("output/prediction/mask_"+file+".jpg", mask_map)
    #position = np.where(out_map>0)
    #boxes = find_boxes(position)
    #print boxes
    #print out_image.shape

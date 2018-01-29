import sys
sys.path.append('/usr/local/bin/')
import os
import random

import cv2
import numpy as np
import multiresolutionimageinterface as mir
from camelyon_dataset import MaskAnnotation

class MRImage:
    def __init__(self, tif_file, xml_file=None, skip_empty=True):
        self.tif_file = tif_file
        self.skip_empty = skip_empty
        reader = mir.MultiResolutionImageReader()
        self.mr_image = reader.open(tif_file)
        self.max_size = self.mr_image.getDimensions()
        self.image_id = int(self.tif_file.split("/")[-1].split("_")[1])/20
        self.ds, self.thr_image = self.binary_img()
        
        if xml_file is not None:
            self.anno = MaskAnnotation(xml_file)

    def binary_img(self):
        if self.image_id ==4:
            level = 7
        else:
            level = 5

        ds = self.mr_image.getLevelDownsample(level)
        shape = self.mr_image.getLevelDimensions(level)
        image = self.mr_image.getUCharPatch(0,0,shape[0], shape[1], level)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.image_id == 4:
            gray_image[gray_image>200] = 255
        else :
            gray_image[gray_image==0] = 255
        blur = cv2.GaussianBlur(gray_image,(5,5),0)
        ret1,binary_image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return ds, binary_image

    def is_empty(self, image_patch):
        #check = image_patch == image_patch[0,0,:]
        return len(np.unique(image_patch))<30
        
    def is_error(self, image_patch):
        return self.image_id == 4 and np.min(image_patch) == 255

    def is_background(self, rect):
        x1 = rect[0]/self.ds
        y1 = rect[1]/self.ds
        x2 = rect[2]/self.ds
        y2 = rect[3]/self.ds
        return 0 not in np.unique(self.thr_image[int(y1):int(y2+0.5), int(x1):int(x2+0.5)])

    def sub_region(self, rect):
        x1,y1,x2,y2 = rect
        
        if self.skip_empty and self.is_background(rect):
            return None
        image_patch = self.mr_image.getUCharPatch(x1, y1, x2-x1+1, y2-y1+1, 0)

        if self.is_error(image_patch):
             reader = mir.MultiResolutionImageReader()
             self.mr_image = reader.open(self.tif_file)
        
        if self.skip_empty and self.is_empty(image_patch):
            return None
        if hasattr(self, 'anno'):
            mask = self.anno.get_mask((x1,y1,x2,y2), 255)
            return image_patch, mask
        return image_patch

if __name__ == '__main__':
    mr_img = MRImage("/opt/intern/users/yuewang/dataset/Camelyon17/images/patient_015_node_1.tif")
    img = mr_img.sub_region([180000,90000,200000,100000])
    print img.type
    cv2.imshow(img)
    cv2.waitKey()

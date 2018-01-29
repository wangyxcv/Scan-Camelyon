
import sys
sys.path.append("/usr/local/bin")
import multiresolutionimageinterface as mir
from matplotlib import pyplot as plt
import numpy as np
from skimage import data,filters
import cv2
import os

r = mir.MultiResolutionImageReader()

path = "/data/yuewang/CAMELYON17/testing/centres/"
id_list = []
for file in os.listdir(path):
    patiend_id = file.split("_")[1]
    mr_image = r.open(path+file)
    level = 7
    shape = mr_image.getLevelDimensions(level)
    img = mr_image.getUCharPatch(0, 0, shape[0], shape[1], level)
    #id_list.append(patiend_id)
    a = np.argmax(np.bincount(img.flatten())) 
    if int(patiend_id) == 148 or int(patiend_id)==122:
        print file
        print a
        #plt.imshow(img)
        #plt.show()

'''
mr_image = r.open("/opt/intern/users/yuewang/dataset/Camelyon17/images/patient_089_node_3.tif")
mask_image = r.open("/opt/intern/users/yuewang/dataset/Camelyon17/gt/patient_089_node_3.tif")

image_level_0 = mr_image.getUCharPatch(95000, 12400, 1204, 1204, 0)
mask_level_0 = mask_image.getUCharPatch(95000, 12400, 1204, 1204, 0)
#print np.bincount(mask_level_0.flatten())

#cv2.imwrite("test_2000.jpg", image_level_0)
#cv2.imwrite("test_mask.jpg", mask_level_0)
image_level_0 = image_level_0[106:1204-106,106:1204-106,:]
mask_level_0 = mask_level_0[106:1204-106,106:1204-106,0]
output_1 = cv2.imread("/opt/intern/users/yuewang/ScanNet-FCN/output/prediction/output_5.jpg")
output_2 = cv2.imread("/opt/intern/users/yuewang/ScanNet-FCN/output/prediction/output_4.jpg")
print image_level_0.shape
print output_1.shape
print output_2.shape
#image = np.zeros((1204-212,1204-212,3), dtype=np.uint8)
#image[:,:,0] = mask_level_0
#image[:,:,1] = output_1[:,:,0]
#output_1[:,:,1:] = image_level_0[:,:,1:]
print np.unique(output_1[:,:,0])
img_out = np.zeros((1204-212,1204-212,3), dtype=np.uint16)
img_out[:] = image_level_0[:]
out = 200.0*output_1[:,:,0]/255 + image_level_0[:,:,0]
out[out>255] = 255
img_out[:,:,0] = out
#img_out[:,:,0] = output_1[:,:,0]
#output_1[output_1>255] = 255
print np.unique(out)
print np.unique(image_level_0[:,:,0])
img_out[:,:,0] = out
plt.subplot(2,2,1)
plt.imshow(img_out)
plt.subplot(2,2,2)
plt.imshow(out)
plt.subplot(2,2,3)
plt.imshow(output_1[:,:,0])
plt.subplot(2,2,4)
plt.imshow(image_level_0[:,:,0])
plt.show()
'''

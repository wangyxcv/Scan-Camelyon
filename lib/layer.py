import _init_path
import caffe
import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy.misc import imresize
import pdb

class SaveImageLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception('it should have only one input')

        params = eval(self.param_str)
        self.save_dir = params["save_dir"]
        self.threhold = params["threhold"]

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        #image = np.zeros_like(bottom[0].shape)
        output = bottom[0].data
        #pdb.set_trace()
        #print "the output image shape is:", output.shape
        #image = np.zeros((242, 242))
        image = output[0][1]
        #image[:,:,1] = output[0][1]
        #image[:,:,2] = output[0][2]
        #print image.shape
        plt.imshow(image, cmap="gray")
        plt.show()
        #image = output[0][0]
        #print image.shape
        #print image
        #print output
        #image = cv2.resize(image, (1204-212,1204-212), interpolation = cv2.INTER_LINEAR)
        #image = imresize(image, [1204-212,1204-212], 'bilinear')
        #for i in range(1,10):
           # image_ = np.where(image>=i*0.1, 255, 0)
            #image = cv2.resize(image, (1204-212,1204-212), interpolation = cv2.INTER_LINEAR)
            #image = np.where(image>=0.5, 255, 0)
            #cv2.imwrite(self.save_dir+"output_"+str(i)+".jpg", image_)

    def backward(self, top, propagate_down, bottom):
        pass

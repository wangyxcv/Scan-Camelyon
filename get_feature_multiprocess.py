import numpy as np
import time
import multiprocessing
import scipy.io as sio
import signal
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--img_list', nargs='?', type=str, help="img")
parser.add_argument('--folder', nargs='?', type=str, help="img folder")
parser.add_argument('--model_def', nargs='?', type=str, help="prototxt")
parser.add_argument('--model_weight', nargs='?', type=str, help="caffemodel")
parser.add_argument('--feature_name', nargs='?', type=str, help="data(top or bottom) name you want to save")
parser.add_argument('--out', nargs='?', type=str, help="output .mat file, len(img_list) * len(feature)")
parser.add_argument('--gpu', nargs='+', type=str, help="gpu list, e.g.  --gpu 1 2 3 4")
parser.add_argument('--env', nargs='+', type=str, help="specify env usage. Append them to sys.path. If you add caffe path in $PYTHONPATH, ignore this option.\n If you don't want to use caffe in $PYTHONPATH, please use -- $ unset PYTHONPATH")

args = parser.parse_args()

env = args.env
for i in env:
   sys.path.append(i)
import caffe


feature_name = args.feature_name
img_file = args.img_list
out = args.out
folder = args.folder
model_def = args.model_def
model_weights = args.model_weight
gpu_list = [int(x) for x in args.gpu]
with open(img_file) as f:
    img_file = [folder + x.strip().split(" ")[0] for x in f]

class CaffeNet:
    def __init__(self, model_def, model_weights, device_id, input_size=None):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)

        self.net = caffe.Net(model_def,
                    model_weights,
                    caffe.TEST)

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        mu = np.array([104.0, 117.0, 123.0], dtype=np.float32)
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', mu)
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2,1,0))

    def get_feature(self, feature_name, filename):
        image = caffe.io.load_image(filename)
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image
        output = self.net.forward()
        prob = output[feature_name][0].flatten()
        print prob
        return prob

def build_net():
    global net
    global exit_flag
    exit_flag = False
    my_id = multiprocessing.current_process()._identity[0]
    if my_id > len(gpu_list):
        exit_flag = True
    else:
        net = CaffeNet(model_def, model_weights, gpu_list[my_id-1])

    global cou

def get_feature(work):
    global net
    global exit_flag
    if exit_flag:
        return
    file_name = work[0]
    work_id = work[1]
    print(str(work_id) + " / " + str(len(img_file)))
    feature = net.get_feature(feature_name, file_name)
    return feature, work_id

def exit_multiprocess(signum, frame):
    global pool
    pool.terminate()
    exit()

signal.signal(signal.SIGINT, exit_multiprocess)
work = zip(img_file, range(len(img_file)))
pool = multiprocessing.Pool(len(gpu_list), initializer=build_net)

signal.signal(signal.SIGINT, exit_multiprocess)

feature_result = pool.map_async(get_feature, work).get()
pool.close()
pool.join()

feature_mat = np.zeros((len(feature_result), len(feature_result[0][0])), dtype=np.float32)
for i in range(len(feature_result)):
    feature_mat[feature_result[i][1], :] = feature_result[i][0]
sio.savemat(out, {'feature': feature_mat})



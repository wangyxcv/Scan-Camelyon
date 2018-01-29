import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'caffe', 'python')
#caffe_path = "/home/haozhang/yjxcaffe_py/caffe/python"
add_path(caffe_path)
#add_path("/home/haozhang/cv-malong-caffe/caffe/python")
# Add lib to PYTHONPATH
lib_path = osp.join(this_dir)
add_path(lib_path)

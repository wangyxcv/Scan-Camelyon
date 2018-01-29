#!/bin/sh

python  get_feature_multiprocess.py \
--gpu 6 \
--model_weight /opt/intern/users/yuewang/ScanNet-FCN/output/model/ScanNet_iter_3000.caffemodel \
--model_def model/test_py.prototxt \
--img_list /opt/intern/users/yuewang/ScanNet-FCN/data/Camelyon17/positive_list.txt \
--folder /opt/intern/users/yuewang/ScanNet-FCN/data/Camelyon17/train_set/ \
--feature_name softmax \
--out output/prediction/output.mat \
--env /opt/intern/users/yuewang/ScanNet-FCN/caffe/python 2>&1| tee output/logs/test_py.txt

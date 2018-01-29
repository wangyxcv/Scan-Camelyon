#!/bin/sh

./caffe/build/tools/caffe.bin test \
--gpu 6 \
--weights /opt/intern/users/yuewang/ScanNet-FCN/output/model/ScanNet_iter_40000.caffemodel \
--model model/test_single.prototxt \
--iterations 901 2>&1| tee output/logs/test.txt

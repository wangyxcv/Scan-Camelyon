#!/bin/sh

./caffe/build/tools/caffe.bin train \
--gpu 2 \
--solver model/solver.prototxt \
--weights output/model/ScanNet_iter_400000.caffemodel 2>&1| tee output/logs/train_1_1.txt

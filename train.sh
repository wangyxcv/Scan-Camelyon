#!/bin/sh

./caffe/build/tools/caffe.bin train \
--gpu 1 \
--solver model/solver.prototxt \
--weights /opt/intern/users/yuewang/ScanNet-FCN/model/pretrained/VGG_ILSVRC_16_layers.caffemodel 2>&1| tee output/logs/train_sample_image_rotate.txt

#!/bin/sh

./caffe/build/tools/caffe.bin train \
--gpu 3 \
--solver model/solver_random_list.prototxt \
--weights model/pretrained/VGG_ILSVRC_16_layers.caffemodel 2>&1| tee output/logs/train_random_list.txt

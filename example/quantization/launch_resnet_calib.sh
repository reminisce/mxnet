#! /bin/sh
python resnet_calib.py --model=imagenet1k-resnet-152 --data-val=./data/val_256_q90.rec --gpus=0 --data-nthreads=60

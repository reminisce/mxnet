#! /bin/sh
python inception_v3_calib.py --gpus=0 --data-nthreads=60 --data-val=./data/val_480_q95.rec

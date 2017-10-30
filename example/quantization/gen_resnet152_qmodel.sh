#!/usr/bin/env bash
python gen_quantized_resnet.py --model=imagenet1k-resnet-152 --gpus=0 --data-nthreads=60
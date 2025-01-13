#!/bin/bash

# DATASET=/path/to/filelist.txt
# RESUME=/path/to/resume.pkl

docker run --gpus all --ipc host -it --rm \
  -w /scratch \
  -e HOME=/scratch \
  -v $(pwd):/scratch \
  -v /media/Datauniverse:/media/Datauniverse \
  -v /raid/alex/data:/raid/alex/data \
  -v /raid/cache/:/raid/cache \
  --user $(id -u):$(id -g) \
  stylegan3 \
  python train.py \
  --cfg=stylegan2 \
  --metrics=none \
  --nobench=False \
  --outdir=./results \
  --data=$DATASET \
  --gpus=4 \
  --snap=20 \
  --gamma=10 \
  --mirror=1 --seed=42 --aug=ada \
  --mbstd-group=4 --batch=32 --batch-gpu=8 --workers=32 --prefetch=16 \
  --resolution=1024 \
  --kimg=40000
#   --resume_kimg=0 \
#   --resume=$RESUME \
#   --glr .001 --dlr .001 \ 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

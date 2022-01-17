docker run --gpus all --shm-size=96g -it --rm \
  -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch -v /media/Datauniverse:/datasets \
  stylegan3 \
  python train.py \
  --outdir=./results --cfg=stylegan3-r --data=/datasets/coral/data/filelists/coral_clean_rel.txt \
  --gpus=4 --batch=32 --workers=16 --metrics=none --snap=10 --gamma=0 --nobench=True \
  --mirror=1 --seed=0 --aug=noaug
#   --mirror=1 --aug=ada --target=.6 --seed=0


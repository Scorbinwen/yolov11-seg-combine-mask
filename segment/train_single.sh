python train.py --pretrained_ckpt ../runs/train18/weights/epoch300.pt --model ../ultralytics/cfg/models/11/yolo11-seg.yaml --data ../ultralytics/cfg/datasets/neiceng_40cls_single.yaml  --device "0" --save_period 10 --epochs 600 --batch 1

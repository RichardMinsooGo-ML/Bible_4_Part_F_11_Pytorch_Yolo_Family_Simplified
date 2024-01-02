# Engilish
*  **Theory** : [https://wikidocs.net/167699](https://wikidocs.net/226339) <br>
*  **Implementation** : [https://wikidocs.net/167693](https://wikidocs.net/226340)

# 한글
*  **Theory** : [https://wikidocs.net/187967](https://wikidocs.net/225897) <br>
*  **Implementation** : [https://wikidocs.net/167666](https://wikidocs.net/226038)

This repository is folked from [https://github.com/yjh0410/RT-ODLab](https://github.com/yjh0410/RT-ODLab).
At this repository, simplification and explanation and will be tested at Colab Environment.

# YOLOv7:

|    Model    |   Backbone    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|---------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv7-Tiny | ELANNet-Tiny  | 8xb16 |  640  |         39.5           |       58.5        |   22.6            |   7.9              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov7_tiny_coco.pth) |
| YOLOv7      | ELANNet-Large | 8xb16 |  640  |         49.5           |       68.8        |   144.6           |   44.0             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov7_coco.pth) |
| YOLOv7-X    | ELANNet-Huge  |       |  640  |                        |                   |                   |                    |  |

- For training, we train `YOLOv7` and `YOLOv7-Tiny` with 300 epochs on 8 GPUs.
- For data augmentation, we use the [YOLOX-style](https://github.com/Megvii-BaseDetection/YOLOX) augmentation including the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation.
- For optimizer, we use `AdamW` with weight decay 0.05 and per image learning rate 0.001 / 64.
- For learning rate scheduler, we use Cosine decay scheduler.
- For YOLOv7's structure, we replace the coupled head with the YOLOX-style decoupled head.
- I think YOLOv7 uses too many training tricks, such as `anchor box`, `AuxiliaryHead`, `RepConv`, `Mosaic9x` and so on, making the picture of YOLO too complicated, which is against the development concept of the YOLO series. Otherwise, why don't we use the DETR series? It's nothing more than doing some acceleration optimization on DETR. Therefore, I was faithful to my own technical aesthetics and realized a cleaner and simpler YOLOv7, but without the blessing of so many tricks, I did not reproduce all the performance, which is a pity.
- I have no more GPUs to train my `YOLOv7-X`.

## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_F_07_Pytorch_Yolov7.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

```Shell
! pip install thop
```

## Step x. Download pretrained weight

```Shell
! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov7_tiny_coco.pth
! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov7_coco.pth
```

## Demo
### Detect with Image
```Shell
# Detect with Image

! python demo.py --mode image \
                 --path_to_img /content/dataset/demo/images/ \
                 --cuda \
                 -m yolov7_tiny \
                 --weight /content/yolov7_tiny_coco.pth \
                 -size 640 \
                 -vt 0.4
                 # --show

# See /content/det_results/demos/image
```

### Detect with Video
```Shell
# Detect with Video

! python demo.py --mode video \
                 --path_to_vid /content/dataset/demo/videos/street.mp4 \
                 --cuda \
                 -m yolov7_tiny \
                 --weight /content/yolov7_tiny_coco.pth \
                 -size 640 \
                 -vt 0.4 \
                 --gif
                 # --show

# See /content/det_results/demos/video Download and check the results
```

### Detect with Camera
```Shell
# Detect with Camera
# it don't work at Colab. Use laptop

# ! python demo.py --mode camera \
#                  --cuda \
#                  -m yolov7_tiny \
#                  --weight /content/yolov7_tiny.pth \
#                  -size 640 \
#                  -vt 0.4 \
#                  --gif
                 # --show
```

## Download COCO Dataset

```Shell
# COCO dataset download and extract

# ! wget http://images.cocodataset.org/zips/train2017.zip
! wget http://images.cocodataset.org/zips/val2017.zip
! wget http://images.cocodataset.org/zips/test2017.zip
# ! wget http://images.cocodataset.org/zips/unlabeled2017.zip

# ! unzip train2017.zip  -d dataset/COCO
! unzip val2017.zip  -d dataset/COCO
! unzip test2017.zip  -d dataset/COCO

# ! unzip unlabeled2017.zip -d dataset/COCO

# ! rm train2017.zip
# ! rm val2017.zip
# ! rm test2017.zip
# ! rm unlabeled2017.zip

! wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

! unzip annotations_trainval2017.zip -d dataset/COCO
# ! unzip stuff_annotations_trainval2017.zip
# ! unzip image_info_test2017.zip
# ! unzip image_info_unlabeled2017.zip

# ! rm annotations_trainval2017.zip
# ! rm stuff_annotations_trainval2017.zip
# ! rm image_info_test2017.zip
# ! rm image_info_unlabeled2017.zip

clear_output()
```

## Test YOLOv7
Taking testing YOLOv7-Tiny on COCO-val as the example,
```Shell
# Test YOLOv7
! python test.py --cuda \
                 -d coco \
                 --data_path /content/dataset \
                 -m yolov7_tiny \
                 --weight /content/yolov7_tiny_coco.pth \
                 -size 640 \
                 -vt 0.4
                 # --show
# See /content/det_results/coco/yolov7
```

## Evaluate YOLOv7
Taking evaluating YOLOv7-Tiny on COCO-val as the example,
```Shell
# Evaluate YOLOv7

! python eval.py --cuda \
                 -d coco-val \
                 --data_path /content/dataset \
                 --weight /content/yolov7_tiny_coco.pth \
                 -m yolov7_tiny
```

# Training test
## Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
```


## Train YOLOv7
### Single GPU
Taking training YOLOv7-Tiny on VOC as the example,
```Shell
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov7_tiny \
                  -bs 16 \
                  --max_epoch 20 \
                  --wp_epoch 1 \
                  --eval_epoch 10 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

```Shell
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov7 \
                  -bs 8 \
                  --max_epoch 20 \
                  --wp_epoch 1 \
                  --eval_epoch 10 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

```Shell
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov7_x \
                  -bs 4 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

### Multi GPU
Taking training YOLOv7-Tiny on VOC as the example,
```Shell
# Cannot test at Colab-Pro + environment

# ! python -m torch.distributed.run --nproc_per_node=8 train.py \
#                                   --cuda \
#                                   -dist \
#                                   -d voc \
#                                   --data_path /content/dataset \
#                                   -m yolov7_tiny \
#                                   -bs 128 \
#                                   -size 640 \
#                                   --wp_epoch 3 \
#                                   --max_epoch 300 \
#                                   --eval_epoch 10 \
#                                   --no_aug_epoch 20 \
#                                   --ema \
#                                   --fp16 \
#                                   --sybn \
#                                   --multi_scale \
#                                   --save_folder weights/
```


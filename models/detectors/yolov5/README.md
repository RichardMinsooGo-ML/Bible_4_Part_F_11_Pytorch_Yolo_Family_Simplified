# Engilish
*  **Theory** : [https://wikidocs.net/167699](https://wikidocs.net/226336) <br>
*  **Implementation** : [https://wikidocs.net/167693](https://wikidocs.net/226337)

# 한글
*  **Theory** : [https://wikidocs.net/187967](https://wikidocs.net/218072) <br>
*  **Implementation** : [https://wikidocs.net/167666](https://wikidocs.net/226041)

This repository is folked from [https://github.com/yjh0410/RT-ODLab](https://github.com/yjh0410/RT-ODLab).
At this repository, simplification and explanation and will be tested at Colab Environment.

# YOLOv5:

|   Model   | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv5-N  | 1xb16 |  640  |         29.8           |       47.1        |   7.7             |   2.4              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_n_coco.pth) |
| YOLOv5-S  | 1xb16 |  640  |         37.8           |       56.5        |   27.1            |   9.0              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_s_coco.pth) |
| YOLOv5-M  | 1xb16 |  640  |         43.5           |       62.5        |   74.3            |   25.4             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_m_coco.pth) |
| YOLOv5-L  | 1xb16 |  640  |         46.7           |       65.5        |   155.6           |   54.2             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_l_coco.pth) |

- For training, we train YOLOv5 series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use SGD with weight decay 0.0005 and base per image lr 0.01 / 64, following the setting of the official YOLOv5.
- For learning rate scheduler, we use linear decay scheduler.
- We use decoupled head in our reproduced YOLOv5, which is different from the official YOLOv5'head.


On the other hand, we are trying to use **AdamW** and larger batch size to train our reproduced YOLOv5. We will update the new results as soon as possible.

|   Model   | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv5-N  | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOv5-T  | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOv5-S  | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOv5-M  | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOv5-L  | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOv5-X  | 8xb16 |  640  |                        |                   |                   |                    |  |

- For training, we train YOLOv5 series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64. We are not good at using SGD.
- For learning rate scheduler, we use linear decay scheduler.
- We use decoupled head in our reproduced YOLOv5, which is different from the official YOLOv5'head.

## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_F_05_Pytorch_Yolov5.git
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
# yolov5 pretrained weight is not working at Colab

# ! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_n_coco.pth
# ! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_s_coco.pth
# ! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_m_coco.pth
# ! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_l_coco.pth
```

## Demo
### Detect with Image
```Shell
# Detect with Image

# ! python demo.py --mode image \
#                  --path_to_img /content/dataset/demo/images/ \
#                  --cuda \
#                  -m yolov5_l \
#                  --weight /content/yolov5_l_coco.pth \
#                  -size 640 \
#                  -vt 0.4
                 # --show

# See /content/det_results/demos/image
```

### Detect with Video
```Shell
# Detect with Video

# ! python demo.py --mode video \
#                  --path_to_vid /content/dataset/demo/videos/street.mp4 \
#                  --cuda -m yolov5_m \
#                  --weight /content/yolov5_m_coco.pth \
#                  -size 640 \
#                  -vt 0.4 \
#                  --gif
                 # --show

# See /content/det_results/demos/video Download and check the results
```

### Detect with Camera
```Shell
# Detect with Camera
# it don't work at Colab. Use laptop

# ! python demo.py --mode camera \
#                  --cuda \
#                  -m yolov5_s \
#                  --weight /content/yolov5_s_coco.pth \
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


## Test YOLOv5
Taking testing YOLOv5 on COCO-val as the example,
```Shell
# Test YOLOv5

# ! python test.py --cuda \
#                  -d coco \
#                  --data_path /content/dataset  \
#                  -m yolov5_m \
#                  --weight /content/yolov5_m_coco.pth \
#                  -size 640 \
#                  -vt 0.4
                 # --show
# See /content/det_results/coco/yolov5
```

## Evaluate YOLOv5
Taking evaluating YOLOv5 on COCO-val as the example,
```Shell
# Evaluate YOLOv5

# ! python eval.py --cuda \
#                  -d coco-val \
#                  --data_path /content/dataset \
#                  --weight /content/yolov5_m_coco.pth \
#                  -m yolov5_m

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


## Train YOLOv5
### Single GPU
Taking training YOLOv5-S on VOC as the example,
```Shell
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov5_n \
                  -bs 32 \
                  --max_epoch 20 \
                  --wp_epoch 1 \
                  --eval_epoch 10 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

### Multi GPU
Taking training YOLOv5 on VOC as the example,
```Shell
# yolov5_t have some bug
# ! python train.py --cuda \
#                   -d voc \
#                   --data_path /content/dataset \
#                   -m yolov5_t \
#                   -bs 32 \
#                   --max_epoch 3 \
#                   --wp_epoch 1 \
#                   --eval_epoch 3 \
#                   --fp16 \
#                   --ema \
#                   --multi_scale
```


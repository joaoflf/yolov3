# YoloV3
---
Implementation of the YoloV3 object detection algorithm from scratch in PyTorch. Unfortunately I cannot get it to converge on a Pascal VOC dataset aggregation 2007-2012 to a loss less than 0.70889 and score an mAP @ 50 IoU of 43.74

### Clone and install requirements
```bash
$ git clone https://github.com/SannaPersson/YOLOv3-PyTorch.git
$ cd yolov3
$ poetry install
```

### Download pretrained weights on Pascal-VOC
Pretrained weights for Pascal-VOC can be downloaded [here](https://www.dropbox.com/s/irvxuk66bcazgx7/yolo.pt?dl=0)

### Download Pascal VOC dataset
Download the preprocessed dataset from [link](https://www.kaggle.com/aladdinpersson/pascal-voc-yolo-works-with-albumentations). Just unzip this in the main directory.
The file structure of the dataset is a folder with images, a folder with corresponding text files containing the bounding boxes and class targets for each image and two csv-files containing the subsets of the data used for training and testing. 

### Training
Edit main.py file to match the setup you want to use. Then run it. Use notebook.ipynb to run your evaluations. Tracking is made with weights and biases.

### Results
| Model                   | mAP @ 50 IoU |
| ----------------------- |:-----------------:|
| YOLOv3 (Pascal VOC) 	  | 43.74             |

The model was evaluated with confidence 0.2 and IOU threshold 0.45 using NMS.

## YOLOv3 paper 
The implementation is based on the following paper:
### An Incremental Improvement 
by Joseph Redmon, Ali Farhadi

#### Abstract
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
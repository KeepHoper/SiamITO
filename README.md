# SiamITO
The soucre code and trained models of SiamITO: A Light Siamese Network for Infrared Tiny Object Tracking

## Demo
 ![img](https://github.com/KeepHoper/SiamITO/blob/master/output/demo.gif)
 
 ## Requirements
 ```
 numpy
 opencv-python
 torch # recommend 1.8.2+cu111
 torchvision
 visdom
 pillow
 ```
 
 ## Usage
 ```
 cd SiamITO/
 pip install -r requirements.txt
 unzip video/video.zip -d video/
 python demo.py --vis
 ```

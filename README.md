1.ctpn-pytorch

my blog:https://blog.csdn.net/fanzonghao/article/details/106302632

2.setup

nms and bbox utils are written in cython, you have to build the library first.
--cd utils/bbox
--sh make.sh
It will generate a nms.so and a bbox.so in current folder.

3.inference

--change you own model_path , dir_path, save_path and save_path_json in inference.py
python inference.py


4.labelme json to txt:

--change you own path in json_txt.py
python json_txt.py


5.train

follow icdar15 dataset format, x1,y1,x2,y2,x3,y3,x4,y4,label,(x1,y1) is left top,(x2,y2) is right top.

image
--|
    1.jpg
    1.txt   

-- Augmentation include color trans, Expand, 
-- change you own model_path, dataset_path in train_v2.py

python train_v2.py


6.reference

 1. https://github.com/eragonruan/text-detection-ctpn
 2. https://github.com/BADBADBADBOY/pytorch.ctpn



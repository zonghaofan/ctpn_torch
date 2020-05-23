# encoding:utf-8
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataloader.data_util import GeneratorEnqueuer
from dataloader.create_random_ctpn_bbox import get_ctpn_bboxs

# DATA_FOLDER = "/src/notebooks/train_data/cwtdata/" #train data root path

# DATA_FOLDER = '/SSD/ctpn/data/icdar2015/ch4_training_images'
# GT_FOLDER = '/SSD/ctpn/data/icdar2015/train_gts'

DATA_FOLDER = '/SSD/ctpn/data/redfile/效果差的_去章'
GT_FOLDER = '/SSD/ctpn/data/redfile/效果差的_去章'
stride_step = 16 #  pooling down
random_angle = 2 # img will transform random from -5 to 5 when training


# def get_training_data():
#     img_files = []
#     exts = ['jpg', 'png', 'jpeg', 'JPG']
#     for parent, dirnames, filenames in os.walk(os.path.join(DATA_FOLDER, "image")):
#         for filename in filenames:
#             for ext in exts:
#                 if filename.endswith(ext):
#                     img_files.append(os.path.join(parent, filename))
#                     break
#     print('Find {} images'.format(len(img_files)))
#     return img_files

def get_training_data():
    img_files = [os.path.join(DATA_FOLDER,i) for i in os.listdir(DATA_FOLDER) if '.jpg' in i]
    print('Find {} images'.format(len(img_files)))
    return img_files

def load_annoataion(bboxs):
    bbox = []
    for line_box in bboxs:
        for item in line_box:
            x_min, y_min, x_max, y_max = map(int, item)
            bbox.append([x_min, y_min, x_max, y_max, 1])
    return bbox


def generator(vis=False):
    image_list = np.array(get_training_data())
    print('{} training images in {}'.format(image_list.shape[0], DATA_FOLDER))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        for i in index:
            try:
                im_fn = image_list[i]
                # _, fn = os.path.split(im_fn)
                # fn, _ = os.path.splitext(fn)
                # txt_fn = os.path.join(DATA_FOLDER, "label", fn + '.txt')
                txt_fn = os.path.join(GT_FOLDER, im_fn.split('/')[-1].split('.')[0]+'.txt')
                # print('====txt_fn',txt_fn)
                if not os.path.exists(txt_fn):
                    print("Ground truth for image {} not exist!".format(im_fn))
                    continue
                bboxs,im,im_info = get_ctpn_bboxs(im_fn, txt_fn, step=stride_step, random_angle=random_angle)
                bbox = load_annoataion(bboxs)
                if len(bbox) == 0:
                    print("Ground truth for image {} empty!".format(im_fn))
                    continue
                yield [im], bbox, im_info

            except Exception as e:
                print(e)
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


def test_batch():
    gen = get_batch(num_workers=2, vis=False)
    image_nums = len(os.listdir(DATA_FOLDER))
    print('===image_nums', image_nums)
    epochs = 1
    for epoch in range(epochs):
        for i in range(image_nums):
            image_ori, bbox, im_info = next(gen)
if __name__ == '__main__':
    test_batch()



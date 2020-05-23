import numpy as np
import logging
# import xml.etree.ElementTree as ET
import cv2
import os

class SubtDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if is_test:
            image_sets_file = self.root
        else:
            image_sets_file = self.root

        self.imgs_list_path = SubtDataset._read_image_ids(image_sets_file)
        # print('====self.ids:', self.ids)
        # self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root

        # if os.path.isfile(label_file_name):
        #     class_string = ""
        #     with open(label_file_name, 'r') as infile:
        #         for line in infile:
        #             class_string += line.rstrip()
        #
        #     # classes should be a comma separated list
        #
        #     classes = class_string.split(',')
        #     # prepend BACKGROUND as first class
        #     classes.insert(0, 'BACKGROUND')
        #     # # #notice pascal voc需要加上
        #     # classes = [elem.replace(" ", "") for elem in classes]
        #     self.class_names = tuple(classes)
        #     print('===len( self.class_names)', len(self.class_names))
        #     logging.info("VOC Labels read from file: " + str(self.class_names))
        #
        # else:
        #     logging.info("No labels file, using default VOC classes.")
        self.class_names = ('BACKGROUND', 'word')
        #
        #
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        print('==self.class_dict', self.class_dict)
    def __getitem__(self, index):
        img_list_path = self.imgs_list_path[index]
        # print('===image_id', image_id)
        image, boxes, labels = self._get_annotation(img_list_path)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.target_transform:
            boxes, labels = self.target_transform(boxes)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    # def get_annotation(self, index):
    #     image_id = self.ids[index]
    #     return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.imgs_list_path)

    @staticmethod
    def _read_image_ids(image_sets_file):
        # print('==image_sets_file:', image_sets_file)
        imgs_list_path = [os.path.join(image_sets_file, i) for i in os.listdir(image_sets_file) if '.jpg' in i]
        return imgs_list_path

    def _get_annotation(self, img_list_path):
        txt_file = img_list_path.replace('.jpg', '.txt')
        img = cv2.imread(img_list_path).astype(np.float32)
        fid = open(txt_file, 'r', encoding='utf-8')
        bboxes = []
        labels = []
        class_name = 'word'
        for line in fid.readlines():
            line = line.strip().replace('\ufeff', '').split(',')
            # print(' line===', line)
            line = line[:8]
            line = [int(x) for x in line]
            line = np.array(line)
            line = line.reshape(4, 2)
            line = cv2.minAreaRect(line)
            line = cv2.boxPoints(line).astype(np.int)
            line = self.order_point(line)
            bboxes.append(line.reshape(-1))
            labels.append(self.class_dict[class_name])
        return img, np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def order_point(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = np.sum(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # the top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

if __name__ == '__main__':
    from torch.utils.data import DataLoader, ConcatDataset
    import torch

    dataset_path = '/red_detection/SSD/ctpn/data/redfile/效果差的_去章'
    datasets = []
    dataset = SubtDataset(dataset_path, transform=None,
                          target_transform=None)
    # dataset.class_names
    # datasets.append(dataset)
    # train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(dataset)))
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    for i, (image, boxes, labels) in enumerate(train_loader):
        # pass
        # if i<1:
            print('==image.shape:', image.shape)
            print('==boxes:', boxes)
            print('===labels:', labels)
            # break


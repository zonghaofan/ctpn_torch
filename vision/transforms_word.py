# from https://github.com/amdegroot/ssd.pytorch


import torch
from torchvision import transforms
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import numpy as np
import types
from numpy import random

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

# 8个点变四个点
def eight_point_four_point(a):
    bboxs = []
    for box in a:
        box = np.array(box).reshape(4, 2)
        x1, y1, x2, y2 = np.min(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 0]), np.max(box[:, 1])
        bboxs.append([x1, y1, x2, y2])
    bboxs = np.array(bboxs)
    return bboxs

# 四个点变8个点
def four_point_eight_point(a):
    bboxs = []
    for box in a:
        x1, y1, x2, y2 = box
        bboxs.append([x1, y1, x2, y1, x2, y2, x1, y2])
    bboxs = np.array(bboxs)
    return bboxs

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class SubtractMeansDivStd(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.std
        # # print(image)
        # # print('===image.shape:', image.shape)
        # image = transforms.ToTensor()(image)
        # image = transforms.Normalize(mean=self.mean, std=self.std)(image)
        return image, boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, max_size=1600, min_size=800):
        self.max_size = max_size
        self.min_size = min_size

    def __call__(self, image, boxes=None, labels=None):
        # image = cv2.resize(image, (self.size,
        #                          self.size))
        boxes, image, _ = self.resize_image_ctpn_bboxs(img=image, max_size=self.max_size, min_size=self.min_size, boxes=boxes)
        return image, boxes, labels

    def resize_image_ctpn_bboxs(self, img, step=16, max_size=1600, min_size=800, boxes=None):
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(min_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        new_h = int(img_size[0] * im_scale)
        new_w = int(img_size[1] * im_scale)

        new_h = new_h if new_h // step == 0 else (new_h // step + 1) * step
        new_w = new_w if new_w // step == 0 else (new_w // step + 1) * step

        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        re_size = re_im.shape
        h, w, c = re_im.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        bboxs = self.get_ctpn_bbox(boxes, img_size, re_size, step)
        bboxs = self.load_annoataion(bboxs)#将每一个小框转换成四点
        return bboxs, re_im, im_info
    def get_ctpn_bbox(self, rects,img_size,re_size,step=16):
        bboxes = []
        for line in rects:
            line = line.reshape(4,2)
            line[:, 0] = line[:, 0] / img_size[1] * re_size[1]
            line[:, 1] = line[:, 1] / img_size[0] * re_size[0]
            if((line[0][0]-line[1][0])==0):
                continue
            if((line[2][0]-line[3][0])==0):
                continue
            k_top, b_top = self.cal_k_b(line[0], line[1])
            k_bottom, b_bottom = self.cal_k_b(line[2], line[3])
            x_start = min(line[0][0], line[3][0])
            x_end = max(line[1][0], line[2][0])
            end_num = int(((x_end - x_start) - (x_end - x_start) % step) // step)
            # angle = math.atan(-k_top) * (180 / math.pi)
            bbox = []
            # if (abs(angle) > min_angle):
            #     end_num = end_num + 1
            start_num = 0
            # if (angle > 20):
            #     start_num = start_num - 1
            for i in range(start_num, end_num):
                y_s = int(k_top * (x_start + (i) * step) + b_top)
                y_e = int(k_bottom * (x_start + (i + 1) * step) + b_bottom)
                bbox.append([int(x_start + i * step), int(y_s), int(x_start + (i + 1) * step)-1, int(y_e)])
            if(len(bbox)==0):
                y_s = int(k_top * (x_start + (0) * step) + b_top)
                y_e = int(k_bottom * (x_start + (0 + 1) * step) + b_bottom)
                bbox.append([int(x_start + 0 * step), int(y_s), int(x_start + (0 + 1) * step)-1, int(y_e)])
            if(bbox[-1][2]<x_end):
                y_s = int(k_top * (x_start + (end_num) * step) + b_top)
                y_e = int(k_bottom * (x_start + (end_num) * step) + b_bottom)
                bbox.append([int(x_end- step), int(y_s), int(x_end)-1, int(y_e)])
            bboxes.append(bbox)
        bboxes = self.check_bbox(bboxes)
        return bboxes

    def check_bbox(self, bboxes):
        new_bboxs = []
        for line in bboxes:
            box = []
            for item in line:
                if (item[0] > 0 and item[1] > 0 and item[2] > 0 and item[3] > 0):
                    if (item[2] > item[0] and item[3] > item[1]):
                        box.append(item)
            new_bboxs.append(box)
        return new_bboxs

    def cal_k_b(self, coord1, coord2):
        k = (coord2[1] - coord1[1]) / (coord2[0] - coord1[0])
        b = coord1[1] - k * coord1[0]
        return k, b

    def load_annoataion(self, bboxs):
        bbox = []
        for line_box in bboxs:
            for item in line_box:
                x_min, y_min, x_max, y_max = map(int, item)
                bbox.append([x_min, y_min, x_max, y_max, 1])
        return bbox


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (None),
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, input_boxes=None, labels=None):
        #boxes: [[102. 315. 133. 348.]
        #       [24.  31.  86.  95.]]
        #fzh add
        boxes = eight_point_four_point(input_boxes)
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                # return image, boxes, labels
                continue

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
                # if height>width:
                #     w = random.uniform(0.1 * width, 0.7*width)
                #     h = random.uniform(0.2 * height, 0.7*height)
                # else:
                #     w = random.uniform(0.2 * width, 0.7*width)
                #     h = random.uniform(0.1 * height, 0.7 * height)

                w = random.uniform(0.7 * width, width)
                h = random.uniform(0.7 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                #四个点变成８个点 fzh add
                output_boxs = four_point_eight_point(current_boxes)
                return current_image, output_boxs, current_labels

class Expand(object):
    def __init__(self, mean=127.):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        cv2.imwrite('./image.jpg', image)
        ratio = random.uniform(1, 1.2)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image
        cv2.imwrite('./expand_image.jpg', image)
        boxes = boxes.copy()
        # print('boxes, left, top:', boxes, left, top)
        #顺时针　左上角开始
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:4] += (int(left), int(top))
        boxes[:, 4:6] += (int(left), int(top))
        boxes[:, -2:] += (int(left), int(top))

        return image, boxes, labels

class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        #[[],
        # []]
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            #fzh add
            boxes = eight_point_four_point(boxes)
            boxes[:, 0::2] = width - boxes[:, 2::-2]
            boxes = four_point_eight_point(boxes)
        return image, boxes, classes

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # BGR
            ConvertColor(current="BGR", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


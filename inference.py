import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
from torchvision.transforms import transforms
from models.ctpn import CTPN_Model
import time
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def compute_time_deco(function):
    def warpper(*args, **kwargs):
        st = time.time()
        res = function(*args, **kwargs)
        print('{}:spend time:{}'.format(function.__name__, time.time() - st))
        return res
    return warpper

def rotate(img,angle):
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
    return img_rotation

class CTPN_Detector:
    def __init__(self, model_path):
        logging.info('=======载入ctpn模型======')
        model_dict = torch.load(model_path)
        # print('====model_dict',model_dict.keys())
        model = CTPN_Model().cuda()
        model.load_state_dict(model_dict)
        self.model = model
    def resize_image(self, img, min_scale=800, max_scale=1600):
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(min_scale) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_scale:
            im_scale = float(max_scale) / float(im_size_max)
        new_h = int(img_size[0] * im_scale)
        new_w = int(img_size[1] * im_scale)

        new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
        new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return re_im, (new_h / img_size[0], new_w / img_size[1])

    def toTensorImage(self, image, is_cuda=True):
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image).unsqueeze(0)
        if (is_cuda is True):
            image = image.cuda()
        return image
    @compute_time_deco
    def detect_single_img(self, img):
        ori_h, ori_w, _ = img.shape
        img_res, (rh, rw) = self.resize_image(img)
        h, w, c = img_res.shape
        # print('===img.shape==:', h, w, c)
        # cv2.imwrite('./img_res.jpg', img_res)
        im_info = np.array([h, w, c]).reshape([1, 3])
        img = self.toTensorImage(img_res)
        #(b, 20, h, w) #(b, 40, h, w)
        pre_score, pre_reg = self.model(img)
        print('==pre_score.shape===', pre_score.shape)
        print('====pre_reg.shape:', pre_reg.shape)
                                    #(b,10,2,h,w)-->(10, h, w,2)-->(10*h*w, 2)
        score = pre_score.reshape((pre_score.shape[0], 10, 2, pre_score.shape[2], pre_score.shape[3])).squeeze(0).permute(0,2,3,1).reshape((-1, 2))
        score = F.softmax(score, dim=1)
        #(10, h, w, 2)
        score = score.reshape((10, pre_reg.shape[2], -1, 2))

        #(h, w, 10, 2)-->(b, h, w, 20)
        pre_score =score.permute(1, 2, 0, 3).reshape(pre_reg.shape[2],pre_reg.shape[3],-1).unsqueeze(0).cpu().detach().numpy()
        #(b, h, w, 40)
        pre_reg =pre_reg.permute(0, 2, 3, 1).cpu().detach().numpy()

        textsegs, _ = proposal_layer(pre_score, pre_reg, im_info)
        scores = textsegs[:, 0]
        textsegs = textsegs[:, 1:5]

        textdetector = TextDetector(DETECT_MODE='O')
        boxes, text_proposals= textdetector.detect(textsegs, scores[:, np.newaxis], img_res.shape[:2])
        boxes = np.array(boxes, dtype=np.int)
        text_proposals = np.array(text_proposals, dtype=np.int)
        # print('===text_proposals.shape:', text_proposals.shape)

        # #还原到原尺寸box
        # news_boxs = []
        # for i, box in enumerate(boxes):
        #     box = box[:8].reshape(-1, 2)
        #     box_temp = np.zeros(box.shape)
        #     box_temp[:, 0] = box[:, 0] / rw
        #     box_temp[:, 1] = box[:, 1] / rh
        #     news_boxs.append(box_temp.reshape(-1).astype(np.int))
        if boxes is not None:
            boxes[:, [0, 2, 4, 6]] = boxes[:,[0,2,4,6]] / rw
            boxes[:, [1, 3, 5, 7]] = boxes[:,[1,3,5,7]] / rh
            boxes[:, [0, 6]] = np.clip(boxes[:, [0, 6]] - 5, 0, ori_w - 1)#x1限制
            boxes[:, [2, 4]] = np.clip(boxes[:, [2, 4]] + 20, 0, ori_w - 1)#x2限制解决右边界压字

            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - 5, 0, ori_h - 1)#y1限制
            boxes[:, [5, 7]] = np.clip(boxes[:, [5, 7]] + 5, 0, ori_h - 1)#y2限制

        news_text_proposals = []
        for item in text_proposals:
            item = item.reshape(-1, 2)
            item_temp = np.zeros(item.shape)
            item_temp[:, 0] = item[:, 0] / rw
            item_temp[:, 1] = item[:, 1] / rh
            news_text_proposals.append(item_temp.reshape(-1).astype(np.int))

        return boxes, np.array(news_text_proposals)

    def show_img(self, save_path, im_file, boxes, text_proposals):
        img_ori = cv2.imread(im_file)
        img_ori_h, img_ori_w, _ = img_ori.shape
        img_res, (rh, rw) = self.resize_image(img_ori)
        img_res_h, img_res_w, _ = img_res.shape
        im_name = im_file.split('/')[-1]

        for item in text_proposals:
            cv2.rectangle(img_ori, (item[0], item[1]),(item[2], item[3]), (255, 0, 0))

        # with open('./1.txt', 'w', encoding='utf-8') as file:
        #     [file.write(','.join(map(str, box.tolist())) + ',0\n') for box in news_boxs]

        # #画框
        for i, box in enumerate(boxes):
            cv2.polylines(img_ori, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255),
                          thickness=2)

        cv2.imwrite(os.path.join(save_path, im_name), img_ori)
        #输出json

    def output_json(self, save_path, im_file, bboxs):
        """
            输入：图片,对应的bbbox左上角顺时针[[x1,y1,x2,y1,x2,y2,x1,y2]]和名字
            输出：labelme json文件
        """
        import json
        img = cv2.imread(im_file)
        h, w, _ = img.shape

        im_name = im_file.split('/')[-1]
        # 对应输出json的格式
        jsonaug = {}
        jsonaug['flags'] = {}
        jsonaug['fillColor'] = [255, 0, 0, 128]
        # jsonaug['shapes']
        jsonaug['imagePath'] = im_name

        jsonaug['imageWidth'] = w
        jsonaug['imageHeight'] = h
        shapes = []
        for i, bbox in enumerate(bboxs):
            # print('==bbox:', bbox)
            # print('type(bbox[0]):', type(bbox[0]))
            temp = {"flags": {},
                    "line_color": None,
                    # "shape_type": "rectangle",
                    "shape_type": "polygon",
                    "fill_color": None,
                    "label": "word"}
            temp['points'] = []
            temp['points'].append([int(bbox[0]), int(bbox[1])])
            temp['points'].append([int(bbox[2]), int(bbox[3])])
            temp['points'].append([int(bbox[4]), int(bbox[5])])
            temp['points'].append([int(bbox[6]), int(bbox[7])])
            shapes.append(temp)
        # print('==shapes:', shapes)
        jsonaug['shapes'] = shapes

        jsonaug['imageData'] = None
        jsonaug['lineColor'] = [0, 255, 0, 128]
        jsonaug['version'] = '3.16.3'

        cv2.imwrite(os.path.join(save_path, im_name), img)

        with open(os.path.join(save_path, im_name.replace('.jpg', '.json')), 'w+') as fp:
            json.dump(jsonaug, fp=fp, ensure_ascii=False, indent=4, separators=(',', ': '))
        return jsonaug

if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    import os
    # dir_path = './test'
    # save_path = './test_out'
    dir_path = '/SSD/ctpn/data/redfile/图片'
    save_path = '/SSD/ctpn/data/redfile/图片_out'
    save_path_json = '/SSD/ctpn/data/redfile/图片_json_out'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_json):
        os.mkdir(save_path_json)

    imgs_list_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]

    # model_path = './models/ctpn_test.pth'
    model_path = './model_save/ctpn_199.pth'
    CTPN = CTPN_Detector(model_path)

    for i, img_list_path in enumerate(imgs_list_path):
        # if i<1:
        #     img_list_path = '/SSD/ctpn/data/redfile/图片/Doc09100001.jpg'
            logging.info('=======图片路径为{}'.format(img_list_path))
            # print('===img_list_path', img_list_path)
            img = cv2.imread(img_list_path)
            boxes, text_proposals = CTPN.detect_single_img(img)
            CTPN.show_img(save_path, img_list_path, boxes, text_proposals)
            # detect_model.output_json(save_path_json, img_list_path, boxes)





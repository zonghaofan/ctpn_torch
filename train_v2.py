import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from torch import optim
import numpy as np
from torchvision.transforms import transforms
# from dataloader.data_provider import get_batch, DATA_FOLDER
from models.loss import ctpn_loss
from models.ctpn import CTPN_Model
from utils.rpn_msr.anchor_target_layer import anchor_target_layer
from vision.data_preprocessing import TrainAugmentation
from vision import config
from torch.utils.data import DataLoader
from vision.subt_dataset_word import SubtDataset
import cv2
# def toTensorImage(image,is_cuda=True):
#     image = transforms.ToTensor()(image)
#     image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image).unsqueeze(0)
#     if(is_cuda is True):
#         image = image.cuda()
#     return image
#
def toTensor(item,is_cuda=True):
    item = torch.Tensor(item)
    if(is_cuda is True):
        item = item.cuda()
    return item

#查看标注的大框
def vis_data_boxs(image, boxes):
    print(image.dtype)
    print('===image.shape', image.shape)
    image = np.transpose(image, (1, 2, 0))
    image = (image * config.std + config.mean)*255.

    img = image.copy()
    # print('===np.max(image)', np.max(image))
    #测试裁减的文本框
    for one_class_boxs in boxes:
        for box in one_class_boxs:
            print('===box',box)
            cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255),
                          thickness=2)
    cv2.imwrite('test_out.jpg', img)

#查看标注的ctpn小框
def vis_data_small_boxs(image, boxes):
    print(image.dtype)
    image = np.transpose(image, (1, 2, 0))
    image = (image * config.std + config.mean)*255.

    img = image.copy()
    # print('===np.max(image)', np.max(image))
    #测试裁减的文本框
    for box in boxes:
        print('====box:',box)
        x1,y1,x2,y2 = box[0].numpy(),box[1].numpy(),box[2].numpy(),box[3].numpy()
        cv2.rectangle(img,(x1,y1),(x2,y2),color=(0,0,255),thickness=2)
    cv2.imwrite('test_out.jpg', img)

def tensor_img(image):
    image = np.transpose(image, (1, 2, 0))
    image = (image * config.std + config.mean) * 255.

    return image

#debug to show
def debug_data_loader():
    train_transform = TrainAugmentation(config.max_size, config.min_size, config.mean, config.std)
    dataset_path = '/red_detection/SSD/ctpn/data/redfile/效果差的_去章'
    dataset = SubtDataset(dataset_path, transform=train_transform,
                          target_transform=None)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    # debug train loader
    for i, (images, boxes, labels) in enumerate(train_loader):
        if i < 1:
            print('==image.shape:', images.shape)
            for j in range(images.shape[0]):
                print('==boxes:', boxes)
                print('===labels:', labels)
                # vis_data_boxs(images[j].numpy(), boxes.numpy())#屏蔽掉TrainAugmentation中的Resize才能使用
                vis_data_small_boxs(images[j].numpy(), boxes)
                break
        break

def train(train_loader, model, critetion, optimizer, epoch, scheduler):
    model.train()
    loss_total_list = []
    loss_cls_list = []
    loss_ver_list = []
    for i, (images, boxes, labels) in enumerate(train_loader):
        # print('images.shape:', images.shape)
        batch_size, c, h, w = images.shape
        # print('====batch_size, c, h, w:', batch_size, c, h, w)
        im_info = np.array([h, w, c]).reshape([1, 3])
        # print('=====im_info', im_info)
        images = images.cuda()
        optimizer.zero_grad()
        score_pre, vertical_pred = model(images)
        score_pre = score_pre.permute(0, 2, 3, 1)
        vertical_pred = vertical_pred.permute(0, 2, 3, 1)
        gt_boxes = np.array(boxes)
        # print('======gt_boxes', gt_boxes)
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(
            tensor_img(images[0].cpu().numpy()), score_pre, gt_boxes, im_info)
        rpn_labels = toTensor(rpn_labels)
        rpn_bbox_targets = toTensor(rpn_bbox_targets)
        rpn_bbox_inside_weights = toTensor(rpn_bbox_inside_weights)
        rpn_bbox_outside_weights = toTensor(rpn_bbox_outside_weights)

        loss_tatal, loss_cls, loss_ver = critetion(score_pre, vertical_pred, rpn_labels, rpn_bbox_targets,
                                                   rpn_bbox_inside_weights, rpn_bbox_outside_weights)

        loss_tatal.backward()
        optimizer.step()

        loss_total_list.append(loss_tatal.item())
        loss_cls_list.append(loss_cls.item())
        loss_ver_list.append(loss_ver.item())

        if (i % 50 == 0):
            print("{}/{}/loss_total:{}-------loss_cls:{}--------loss_ver:{}---------Lr:{}".format(epoch,i,loss_tatal.item(),loss_cls.item(),loss_ver.item(),scheduler.get_lr()[0]))
    print("********epoch_loss_total:{}********epoch_loss_cls:{}********epoch_loss_ver:{}********Lr:{}".format(
        np.mean(loss_total_list), np.mean(loss_cls_list), np.mean(loss_ver_list), scheduler.get_lr()[0]))
    if not os.path.exists('./model_save'):
        os.mkdir('./model_save')
    torch.save(model.state_dict(), './model_save/ctpn_' + str(epoch) + '.pth')

def main():
    epochs = 200

    train_transform = TrainAugmentation(config.max_size, config.min_size, config.mean, config.std)
    # dataset_path = '/SSD/ctpn/data/redfile/效果差的_去章'
    dataset_path = '/red_detection/SSD/ctpn/data/redfile/标注好ctpn数据'
    dataset = SubtDataset(dataset_path, transform=train_transform,
                          target_transform=None)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    model = CTPN_Model().cuda()

    model_path = './models/ctpn_199.pth'
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)

    critetion = ctpn_loss(sigma=9)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    for epoch in range(epochs):
        scheduler.step()
        train(train_loader, model, critetion, optimizer, epoch, scheduler)

if __name__ == '__main__':
    # debug_data_loader()
    main()

import os
import json
import cv2
import numpy as np

#校正顺时针的四个点　从左上角开始
def cal_stand_points(points):
    rect = np.zeros((4, 2),dtype=np.int)
    s = np.sum(points, axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    # the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    d = np.diff(points, axis=1)
    rect[1] = points[np.argmin(d)]
    rect[3] = points[np.argmax(d)]
    return rect
def lableme_json_txt():
    # path ='./效果差的_去章'
    path ='./标注好ctpn数据'

    imgs_list_path = [os.path.join(path, i) for i in os.listdir(path) if '.jpg' in i]
    print('==len(imgs_list_path)', len(imgs_list_path))
    for i, img_list_path in enumerate(imgs_list_path):
        # if i<1:
            json_list_path = img_list_path.replace('.jpg', '.json')
            output_txt_path = img_list_path.replace('.jpg', '.txt')
            with open(json_list_path, 'r') as file:
                json_info = json.load(file)
            print('===json_info', json_info)
            shapes = json_info['shapes']
            output_points = []
            for shape in shapes:
                points = np.array(shape['points']).astype(np.int)
                points = cal_stand_points(points)
                print('===points', points)
                output_points.append(list(map(str, (points.reshape(-1).tolist()))))
            print('===output_points', output_points)
            with open(output_txt_path, 'w', encoding='utf-8') as file:
                [file.write(','.join(out) + ',###\n') for out in output_points]

if __name__ == '__main__':
    lableme_json_txt()
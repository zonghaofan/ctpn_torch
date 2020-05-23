import glob
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans, avg_iou

# ANNOTATIONS_PATH = "./data/pascalvoc07-annotations"
ANNOTATIONS_PATH = "./效果差的_去章"
CLUSTERS = 9
# 相对原图是否归一化
BBOX_NORMALIZE = True


def show_cluster(data, cluster, max_points=2000):
    '''
    Display bouding box's size distribution and anchor generated in scatter.
    '''
    if len(data) > max_points:
        idx = np.random.choice(len(data), max_points)
        data = data[idx]
    plt.scatter(data[:, 0], data[:, 1], s=5, c='lavender')
    plt.scatter(cluster[:, 0], cluster[:, 1], c='red', s=100, marker="^")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Bounding and anchor distribution")
    plt.savefig("cluster.png")
    plt.show()


def show_width_height(data, cluster, bins=50):
    '''
    Display bouding box distribution with histgram.
    '''
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    width = data[:, 0]
    height = data[:, 1]
    ratio = height / width

    plt.figure(1, figsize=(20, 6))
    plt.subplot(131)
    plt.hist(width, bins=bins, color='green')
    plt.xlabel('width')
    plt.ylabel('number')
    plt.title('Distribution of Width')

    plt.subplot(132)
    plt.hist(height, bins=bins, color='blue')
    plt.xlabel('Height')
    plt.ylabel('Number')
    plt.title('Distribution of Height')

    plt.subplot(133)
    plt.hist(ratio, bins=bins, color='magenta')
    plt.xlabel('Height / Width')
    plt.ylabel('number')
    plt.title('Distribution of aspect ratio(Height / Width)')
    plt.savefig("shape-distribution.png")
    plt.show()


def sort_cluster(cluster):
    '''
    Sort the cluster to with area small to big.
    '''
    if cluster.dtype != np.float32:
        cluster = cluster.astype(np.float32)
    area = cluster[:, 0] * cluster[:, 1]
    cluster = cluster[area.argsort()]
    ratio = cluster[:, 1:2] / cluster[:, 0:1]
    return np.concatenate([cluster, ratio], axis=-1)


# def load_dataset(path, normalized=True):
#     '''
#     load dataset from pasvoc formatl xml files
#     return [[w,h],[w,h]]
#     '''
#     dataset = []
#     for xml_file in glob.glob("{}/*xml".format(path)):
#         tree = ET.parse(xml_file)
#
#         height = int(tree.findtext("./size/height"))
#         width = int(tree.findtext("./size/width"))
#
#         for obj in tree.iter("object"):
#             if normalized:
#                 xmin = int(obj.findtext("bndbox/xmin")) / float(width)
#                 ymin = int(obj.findtext("bndbox/ymin")) / float(height)
#                 xmax = int(obj.findtext("bndbox/xmax")) / float(width)
#                 ymax = int(obj.findtext("bndbox/ymax")) / float(height)
#             else:
#                 xmin = int(obj.findtext("bndbox/xmin"))
#                 ymin = int(obj.findtext("bndbox/ymin"))
#                 xmax = int(obj.findtext("bndbox/xmax"))
#                 ymax = int(obj.findtext("bndbox/ymax"))
#             if (xmax - xmin) == 0 or (ymax - ymin) == 0:
#                 continue  # to avoid divded by zero error.
#             dataset.append([xmax - xmin, ymax - ymin])
#
#     return np.array(dataset)
#
# def load_dataset(path, normalized=True):
#     '''
#     load dataset from pasvoc formatl xml files
#     return [[w,h],[w,h]]
#     '''
#     dataset = []
#     names = [i for i in os.listdir(path) if 'txt' in i]  # args.input_annotation_txt_dir)
#     # print('names:', names)
#     # # 标注的框的宽和高
#     # annotations_w_h = []
#     for name in names:
#         txt_path = os.path.join(path, name)
#         img_path = txt_path.replace('.txt', '.jpg')
#         img = cv2.imread(img_path)
#         img_h, img_w, _ = img.shape
#         # 读取txt文件中的每一行
#         f = open(txt_path, 'r')
#         for line in f.readlines():
#             line = line.rstrip('\n')
#             w, h = line.split(' ')[3:]  # 这时读到的w,h是字符串类型
#             # eval()函数用来将字符串转换为数值型
#             if normalized:
#                 dataset.append((eval(w), eval(h)))
#             else:
#                 dataset.append((eval(w) * 200, eval(h) * 1800))
#         f.close()
#
#     return np.array(dataset)

def resize_image(img, min_scale=800, max_scale=1200):
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

def load_dataset(path, normalized=True):
    '''
    load dataset from pasvoc formatl xml files
    return [[w,h],[w,h]]
    '''
    dataset = []
    names = [i for i in os.listdir(path) if 'txt' in i]  # args.input_annotation_txt_dir)
    # print('names:', names)
    # # 标注的框的宽和高
    # annotations_w_h = []
    for name in names:
        txt_path = os.path.join(path, name)
        img_path = txt_path.replace('.txt', '.jpg')
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        img_resize, _ = resize_image(img)
        res_img_h, res_img_w, _ =img_resize.shape
        # 读取txt文件中的每一行
        f = open(txt_path, 'r')
        for line in f.readlines():
            line = line.rstrip('\n')
            print('===line', line)
            x1, y1,_,_ ,x2, y2,_,_ = map(float, line.split(',')[:-1])
            # w, h = line.split(' ')[3:]  # 这时读到的w,h是字符串类型
            # eval()函数用来将字符串转换为数值型
            w, h = x2/img_w*res_img_w - x1/img_w*res_img_w, y2/img_h*res_img_h-y1/img_h*res_img_h
            if normalized:
                dataset.append((w, h))
            else:
                dataset.append((w, h ))
        f.close()

    return np.array(dataset)


# print("Start to load data annotations on: %s" % ANNOTATIONS_PATH)
# [[w, h], [w, h]]
data = load_dataset(path='./效果差的_去章', normalized=BBOX_NORMALIZE)
# print(data[:3])
# print("Start to do kmeans, please wait for a moment.")
out = kmeans(data, k=CLUSTERS)
print('==out', out)
# out_sorted = sort_cluster(out)
# print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
# #
# show_cluster(data, out, max_points=2000)
#
# if out.dtype != np.float32:
#     out = out.astype(np.float32)
#
# print("Recommanded aspect ratios(width/height)")
# print("Width    Height   Height/Width")
# for i in range(len(out_sorted)):
#     print("%.3f      %.3f     %.1f" % (out_sorted[i, 0], out_sorted[i, 1], out_sorted[i, 2]))
show_width_height(data, out, bins=50)
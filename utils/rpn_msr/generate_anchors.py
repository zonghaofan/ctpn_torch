import numpy as np


def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors

def scale_anchor(anchor, h, w):
    """
    anchor :(0,0,15,15)
    h,w anchor的高宽
"""
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    # print('===x_ctr,y_ctr', x_ctr, y_ctr)
    # print('h,w:', h, w)
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    # print('====scaled_anchor:', scaled_anchor)
    return scaled_anchor


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    # heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]#原先的
    heights = [11, 16, 23, 28, 33, 38, 48, 55, 68, 75]#针对红头文件
    widths = [16]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    return generate_basic_anchors(sizes)


if __name__ == '__main__':
    import time

    t = time.time()
    anchors_ = generate_anchors()
    print(time.time() - t)

    print('====anchors_:', anchors_)
    print('==anchors_.shape:', anchors_.shape)
    # from IPython import embed;
    #
    # embed()

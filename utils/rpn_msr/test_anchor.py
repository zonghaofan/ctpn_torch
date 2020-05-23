import numpy as np
width = 10
height = 8
_feat_stride=[16, ]

shift_x = np.arange(0, width) * _feat_stride
print('shift_x:', shift_x)
shift_y = np.arange(0, height) * _feat_stride
print('shift_y:', shift_y)

shift_x, shift_y = np.meshgrid(shift_x, shift_y)
print('shift_x:', shift_x)
print('shift_y:', shift_y)
# print('===shift_x.shape:', shift_x.shape)
# print('===shift_y.shape:', shift_y.shape)
# print('shift_x.ravel():',shift_x.ravel())
# print('shift_y.ravel():', shift_y.ravel())
# print('np.vstack', np.vstack((shift_x.ravel(), shift_y.ravel(),
#                         shift_x.ravel(), shift_y.ravel())))

shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                    shift_x.ravel(), shift_y.ravel())).transpose()
K = shifts.shape[0]

print('==shifts:', shifts)
print('==shifts.shape', shifts.shape)
print('==K = ', K)

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2])
y = np.array([0, 1])

shift_x, shift_y = np.meshgrid(x, y)
print('==shift_x:', shift_x)
print('==shift_y:', shift_y)
shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
print('==shifts:', shifts)
K = shifts.shape[0]
res = shifts.reshape((1, K, 4)).transpose((1, 0, 2))#k,1,4
print('res.shape:', res.shape)
print('res:', res)
#
# plt.plot(X, Y,
#          color='red',  # 全部点设置为红色
#          marker='.',  # 点的形状为圆点
#          linestyle='')  # 线型为空，也即点与点之间不用线连接
# plt.xlim(-1, 3)
# plt.ylim(-1, 2)
# plt.grid(True)
# plt.show()

b=np.where([[0, 1],
            [1, 0]])
print('==b:', b)
x = np.arange(9.).reshape(3, 3)
print('==x:', x)
index_y, index_x = np.where(x > 3)

print('==index_y,index_x', index_y, index_x)
#同时满足两个条件的
index = np.where((x[:, 0] > 3) & (x[:, 1] < 8))[0]
print('==index', index)
print('x[index]:', x[index])

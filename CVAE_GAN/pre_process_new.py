from PIL import Image
import numpy as np
import os
import cv2

def load_file(file_name='./colormap/'):
    path = file_name
    path_list = os.listdir(path)
    # path_list.sort(key=lambda x: int(x[5:-4]))
    path_list.sort(key=lambda x: int(x[5:-4]))
    out = []
    for i in path_list:
        out.append(str(file_name + i))
    return out


file_num = 0
np_lst = []
file_num=0
for file in load_file():
    L_path = file
    mg_cg = cv2.imread(file)
    mg_cg = cv2.resize(mg_cg,(50,50))
    mg_cg = mg_cg

    np_lst.append(mg_cg)
    print(file,mg_cg.shape)
    file_num+=1
    if file_num==10:break
np_data = np.array(np_lst)
print(np_data.shape)
#     print(out.size)
#     # print(img.shape)  # 高 宽 三原色分为三个二维矩阵
#     # print(img)
#
#     start_point_x = start_point_y = 30
#     sum = 0
#     print(file_num)
#
#     while start_point_x < img.shape[0]: #<279
#         for i in range(9):
#             np_list[file_num][i][sum] = img[start_point_x, start_point_y]
#             # print(img[start_point_x, start_point_y])
#             start_point_y += 63
#             if start_point_y<img.shape[0]:break
#
#         start_point_x += 63
#         start_point_y = 0
#         sum += 1
#         print(sum)
#     file_num += 1
#
# c = np.full(shape=[996, 16, 12, 3], fill_value=255)
# # c = np.zeros([996, 16, 10, 3])
#
# for i in range(996):
#     for ii in range(16):
#         for iii in range(9):
#             c[i][ii][iii] = np_list[i][iii][ii]
print('saving data:{}'.format(np_data.shape))
np.save("img.npy", np_data)
# b = np.load("filename.npy")

# 20220515: 用于单张mask取bboxs，并保存于txt。

import os
import cv2
import numpy as np
from glob import glob
from matplotlib import pyplot as plt


def Dataset_IMG(img_paths):
    Img = np.asarray(glob(img_paths + arge.img_type))
    return Img


def Load_IMG(img_path):
    img = cv2.imread(img_path)  # 读取图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 变为灰度图
    if arge.On_plt:
        plt.subplot(1, 2, 1)
        plt.imshow(gray)
        plt.title('original')
        # plt.show()
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  ## 阈值分割得到二值化图片
    return binary


def Component_IMG(binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    stats = stats[stats[:, 4].argsort()]  # argsort()用法，表示对数据进行从小到大进行排序，返回数据的索引值
    bboxs = stats[:-1, :4]
    return bboxs


def Save_bboxs2txt(bboxs, save_path, img_path):
    num = 0  # FasterRcnn 的label从0开始？
    save_path = os.path.join(save_path, img_path.split('\\')[1].split('.')[0] + '.txt')
    if not os.path.isfile(save_path):
        fw = open(save_path, 'w')
    else:
        fw = open(save_path, 'w+')
    for wl in bboxs:
        x, y, w, h = wl
        txt = str(num) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
        fw.write(txt)
    fw.close()
    print(f'Saved:{save_path}')


def Rectangle_IMG(bboxs, binary):
    color = (255, 0, 0)  # Red color in BGR；红色：rgb(255,0,0)
    thickness = 10
    mask_BGR = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    plt_On = 0
    Rec_num = 0
    for b in bboxs:
        x0, y0 = b[0], b[1]
        x1 = b[0] + b[2]
        y1 = b[1] + b[3]
        start_point, end_point = (x0, y0), (x1, y1)
        if Rec_num == 0:
            plt_On = 1
            mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
        else:
            mask_bboxs = cv2.rectangle(mask_bboxs, start_point, end_point, color, thickness)
    if plt_On == 1:
        plt.subplot(1, 2, 2)
        plt.imshow(mask_bboxs)
        plt.title('mask_bboxs')
        plt.show()


class Config:
    img_paths = "./"     #图像所在的文件夹。
    img_type = '*.png'  #图像格式
    save_path = "./"    #label保存的文件夹。
    On_plt = True

arge = Config()

if __name__ == '__main__':
    print('--------------Start----------------')
    img_loder = Dataset_IMG(arge.img_paths)
    Total_num = img_loder.shape[0]
    Now_num = 0
    for img_path in img_loder:
        Now_num += 1
        print(f'Num:{Now_num}/{Total_num}')
        binary = Load_IMG(img_path)
        bboxs = Component_IMG(binary)
        Save_bboxs2txt(bboxs, arge.save_path, img_path)
        if arge.On_plt:
            Rectangle_IMG(bboxs, binary)
    print('--------------End----------------')

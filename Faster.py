# Faster-R CNN
# Conv Layers 卷积神经网络用于提取特征，得到feature map。
# RPN网络，RPN：全称 Region Proposal Network，负责产生候选区域 (rois)，每张图大概给出 2000 个候选框.用于提取Region of Interests(RoI)。
# RoI pooling, 用于综合RoI和feature map, 得到固定大小的resize后的feature。
# classifier, 用于分类RoI属于哪个类别。
# FasterRCNN的原理以及代码(https://github.com/chenyuntc/simple-faster-rcnn-pytorch)

# 原始图像 ---> 特征提取 ------>RPN 产生候选框 ------> 对候选框进行分类和回归微调。
import itertools
import os
import random
import time
import glob
import six
from tqdm import tqdm
import torch
import ipdb
import matplotlib
from pprint import pprint
import numpy as np
from PIL import Image
from torch.utils import data as data_
from collections import namedtuple, defaultdict
from torchvision.ops import nms
from torchvision.models import vgg16
from torchvision.ops import RoIPool
from torch.nn import functional as F
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from torchnet.meter import ConfusionMeter, AverageValueMeter

# import resource
#
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))



# matplotlib.use('TkAgg')
VOC_BBOX_LABEL_NAMES = ('Tampered',)
print(f'cuda:{torch.cuda.is_available()}')


class Config:
    # data
    TrainData_path = 'data_face/train/img/',
    TrainLabel_path = 'data_face/train/label/',
    TestData_path = 'data_face/test/img/',
    TestLabel_path = 'data_face/test/label/',
    min_size = 600  # image resize
    max_size = 1000  # image resize

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # visualization
    # env = 'faster-rcnn'  # visdom env
    # port = 8097
    plt_train = 10  # vis every N iter
    plt_test = 20

    # preset
    # data = 'voc'
    # pretrained_model = 'vgg16'

    # training
    epoch = 100

    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    pretrain_path = None
    vgg_pretrain = True  # use caffe pretrained model instead of torchvision
    vgg_pretrain_path = 'pretrain_vgg16/vgg16-397923af.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
# 类别和名字对应的列表
BBOX_LABEL_NAMES = ('tampering', 'non-tampering')


# 1.预处理     # URL:http://giantpandacv.com/academic/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E5%8F%8A%E8%B7%9F%E8%B8%AA/%E7%BB%8F%E5%85%B8%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95%E4%BB%A3%E7%A0%81%E8%A7%A3%E6%9E%90/FasterRCNN/%E3%80%90Faster%20R-CNN%E3%80%912.%20Faster%20RCNN%E4%BB%A3%E7%A0%81%E8%A7%A3%E6%9E%90%E7%AC%AC%E4%B8%80%E5%BC%B9/
def read_img(path, dtype=np.float32, color=True):
    """
    读取图像
    :param path:
    :param dtype:
    :param color:
    :return: numpy.ndarray: An image
    """
    f = Image.open(path)

    if color:
        img = f.convert('RGB')
    else:
        img = f.convert('P')

    img = np.asarray(img, dtype=dtype)
    if img.ndim == 2:
        return img[np.newaxis]  # reshape (H, W) -> (1, H, W)
    else:
        return img.transpose((2, 0, 1))  # transpose (H, W, C) -> (C, H, W)


def resize_bbox(bbox, in_size, out_size):
    """
    根据图像缩放尺寸对bbox进行缩放
    :param bbox:
    :param in_size: 图像缩放前尺寸(H,W)
    :param out_size: 图像缩放后尺寸(H_,W_)
    :return: 缩放后bounding boxes
    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """
    根据图像flip情况对bbox进行翻转
    :param bbox:
    :param size:图像缩放后尺寸
    :param y_flip: flip 情况
    :param x_flip:
    :return:
    """

    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def random_flip(img, y_random=False, x_random=False, return_param=False):
    """
    对图像随机选择flip
    :param img: resized(img)
    :param y_random: F or T
    :param x_random:
    :param return_param: img and y/x flip
    :return: flip_or_no(img)
    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]  # ::-1为逆序输出，相当于翻转
    if x_flip:
        img = img[:, :, ::-1]

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


def pytorch_normalize(img):
    """
    对图像归一化处理
    :param img: 0-1
    :return: Tensor 图像
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化处理
                                std=[0.229, 0.224, 0.225])
    # nddarry->Tensor
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size=600, max_size=1000):
    """
    对图像进行缩放，最终长边<=1000 或者 短边<=600
    bounding boxes 同等尺度缩放
    :param img: 0-255
    :param min_size: 600
    :param max_size: 1000
    :return: 缩放后图像
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)  # 选择最小比例，保证长宽都能缩放到规定尺寸(<=)

    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    normalize = pytorch_normalize  # resize到（H * scale, W * scale）大小，anti_aliasing为是否采用高斯滤波
    return normalize(img)


class Transform(object):  # 定义为一个可调用的对象，先赋值，再调用时返回__call__内容
    """
    对图像增强变换
    :param object:
    :return: 变换后的img
    """

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, H_, W_ = img.shape
        scale = H_ / H
        bbox = resize_bbox(bbox, (H, W), (H_, W_))

        img, params = random_flip(img,
                                  x_random=True,
                                  return_param=True)
        bbox = flip_bbox(bbox, (H_, W_),
                         x_flip=params['x_flip'])
        return img, bbox, label, scale


class Dataset:
    """
    生成训练集
    """

    def __init__(self, opt):
        self.opt = opt
        self.LoadDataset = LoadDataset(opt.TrainData_path, opt.TrainLabel_path)
        self.Tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, index):
        # 调用loaddataset中的get_example()从数据集存储路径中将img, bbox, label 一个个的获取出来
        org_img, bbox, label = self.LoadDataset.get_example(index)
        # 调用前面的Transform函数，将图像、label进行最小值最大值放缩归一化
        # 重新调整bboxes的大小，然后随机反转，最后将数据集返回
        img, bbox, label, scale = self.Tsf((org_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.LoadDataset)


class TestDataset:
    def __init__(self, opt):
        self.opt = opt
        self.LoadDataset = LoadDataset(opt.TestData_path, opt.TestLabel_path)

    def __getitem__(self, index):
        org_img, bbox, label = self.LoadDataset.get_example(index)
        img = preprocess(org_img)
        # temp_plt = plt.imshow(org_img.transpose(1,2,0).astype(np.uint8))
        # for y1, x1, y2, x2 in bbox:
        #     temp_plt.axes.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
        # for y1, x1, y2, x2 in gt_bboxes[0]:
        #     temp_plt.axes.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='yellow', linewidth=2))
        # plt.show()
        return img, org_img, org_img.shape[1:], bbox, label

    def __len__(self):
        return len(self.LoadDataset)


class LoadDataset:
    def __init__(self, data_path, label_path):
        self.data_path = glob.glob(data_path[0] + '*.jpg')
        self.label_path = glob.glob(label_path[0] + '*.txt')
        self.label_names = BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.data_path)

    def get_example(self, i):
        img = read_img(self.data_path[i], color=True)
        label = list()
        bbox = list()
        f = open(self.label_path[i])
        img_s = img.copy()
        # plt_img = plt.imshow(np.asarray(img_s.transpose(1, 2, 0), dtype=np.uint8))
        for temp in f:
            class_num, x, y, w, h = [int(i) for i in temp.split()]
            label.append(class_num)
            bbox.append((y, x, y + h, x + w))  # ('ymin', 'xmin', 'ymax', 'xmax')
        #     plt_img.axes.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
        # plt.show()
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        return img, bbox, label

    __getitem__ = get_example


# 2.RPN  # URL:http://giantpandacv.com/academic/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E5%8F%8A%E8%B7%9F%E8%B8%AA/%E7%BB%8F%E5%85%B8%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95%E4%BB%A3%E7%A0%81%E8%A7%A3%E6%9E%90/FasterRCNN/%E3%80%90Faster%20R-CNN%E3%80%913.%20Faster%20RCNN%E4%BB%A3%E7%A0%81%E8%A7%A3%E6%9E%90%E7%AC%AC%E4%BA%8C%E5%BC%B9/

def loc2bbox(src_bbox, loc):
    """
    已知源bbox和位置偏差dx,dy,dh,dw.求目标框G。
    :param src_bbox: （R，4），R为bbox个数，4为左上角和右下角四个坐标
    :param loc:（R，4N），R为bbox个数，4N为位置偏差dx,dy,dh,dw。
    :return:目标框G（R，4N）
    """
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    # src_height为Ph,src_width为Pw，
    # 中心点坐标：(src_ctr_y为Py，src_ctr_x为Px)
    src_height = src_bbox[:, 2] - src_bbox[:, 0]  # ymax-ymin
    src_width = src_bbox[:, 3] - src_bbox[:, 1]  # xmax-xmin
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height  # ymin+0.5h
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width  # xmin+0.5*w

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    # RCNN中提出的边框回归：寻找原始proposal与近似目标框G之间的映射关系.
    # 中心点的偏移，实际y偏移量=偏移量dy *bbox的height
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]  # ctr_y=Gy
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]  # ctr_x = Gx
    h = np.exp(dh) * src_height[:, np.newaxis]  # h为Gh
    w = np.exp(dw) * src_width[:, np.newaxis]  # w为Gw
    # 上面四行得到了回归后的目标框（Gx,Gy,Gh,Gw）

    # 由中心点转换为左上角和右下角坐标
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """
    通过源框和目标框求出位置偏差。
    :param src_bbox: (R,4) R个(ymin,xmin,ymax,xmax)框
    :param dst_bbox:
    :return: loc = dy,dx,dh,dw
    """
    # 计算源框中心点坐标
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width
    # 计算出目标框中心点坐标
    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width
    # 求出最小的正数
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)
    # 根据上面的公式二计算dx，dy，dh，dw
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)
    # if str(dw[0]) == 'nan':
    #     a = 1
    # print(f'dy:{dy}\n',
    #       f'dx:{dx}\n',
    #       f'dh:{dh}\n',
    #       f'dw:{dw}\n')
    # np.vstack按照行的顺序把数组给堆叠起来
    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    """
    计算两个bbox的交并比。
    :param bbox_a:  (R,(ymin,xmin,ymax,xmax))
    :param bbox_b: (R,(ymin,xmin,ymax,xmax))
    :return: IOU
    """
    # top left
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    # l为交叉部分框左上角坐标最大值，为了利用numpy的广播性质，
    # bbox_a[:, None, :2]的shape是(N,1,2)，bbox_b[:, :2]
    # shape是(K,2),由numpy的广播性质，两个数组shape都变成(N,K,2)，
    # 也就是对a里每个bbox都分别和b里的每个bbox求左上角点坐标最大值
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    # br为交叉部分框右下角坐标最小值
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    # 所有坐标轴上tl<br时，返回数组元素的乘积(y1max-yimin)X(x1max-x1min)，
    # bboxa与bboxb相交区域的面积
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    # 计算bboxa的面积
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    # 计算bboxb的面积
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    # 计算IOU
    return area_i / (area_a[:, None] + area_b - area_i)


# 2.1生成anchor
# 缩放后图像为600*800左右，因此选择基准长度16，16*32最大512的区域，16*8最小128合适
def one_point_generate_anchor(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """
    由一个点生成9个anchor框的左上右下4个坐标。
    :param base_size: 基准长度
    :param ratios: 每个点的框形状
    :param anchor_scales: 基准长度的缩放倍数
    :return: 9个框的坐标
    """
    py = base_size / 2.
    px = base_size / 2.
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            # 生成9种不同比例的h和w
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            index = i * len(anchor_scales) + j
            # 计算出anchor_base画的9个框的左上角和右下角的4个anchor坐标值
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


####################################################################################
def all_point_generate_anchor(anchor_base, feat_stride, height, width):
    """
    featmap的每个点都生成9个anchor。
    :param anchor_base: (9,4) 9个框及坐标
    :param feat_stride: img到featmap的缩放比例。eg:vgg16使img小了16倍
    :param height:featmap的高
    :param width:featmap的宽
    :return: featmap.shape*9个anchor(2W+)
    """
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    # shift_x = [[0，16，32，..],[0，16，32，..],[0，16，32，..]...],
    # shift_y = [[0，0，0，..],[16，16，16，..],[32，32，32，..]...],
    # 就是形成了一个纵横向偏移量的矩阵，也就是特征图的每一点都能够通过这个
    # 矩阵找到映射在--原图中--的具体位置!
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 经过刚才的变化，其实大X,Y的元素个数已经相同，看矩阵的结构也能看出，
    # 矩阵大小是相同的，X.ravel()之后变成一行，此时shift_x,shift_y的元
    # 素个数是相同的，都等于特征图的长宽的乘积(像素点个数)，不同的是此时
    # 的shift_x里面装得是横向看的x的一行一行的偏移坐标，而此时的y里面装
    # 的是对应的纵向的偏移坐标！
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    A = anchor_base.shape[0]
    # 读取特征图中元素的总个数
    K = shift.shape[0]
    # 用基础的9个anchor的坐标分别和偏移量相加，最后得出了所有的anchor的坐标，
    # 四列可以堪称是左上角的坐标和右下角的坐标加偏移量的同步执行，飞速的从
    # 上往下捋一遍，所有的anchor就都出来了！一共K个特征点，每一个有A(9)个
    # 基本的anchor，所以最后reshape((K*A),4)的形式，也就得到了最后的所有
    # 的anchor左上角和右下角坐标.
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    # 为什么需要将特征图对应回原图呢？这是因为我们要框住的目标是在原图上，而我们选
    # Anchor是在特征图上，Pooling
    # 之后特征之间的相对位置不变，但是尺寸已经减少为了原始图的
    # 1/16，而我们的Anchor是为了框住原图上的目标而非特征图上的，所以注意一下
    # Anchor一定指的是针对原图的，而非特征图。
    return anchor


def _get_inside_index(anchor, H, W):
    """
    保留在图像内部的anchor
    :param anchor: 全部anchor
    :param H: img.H
    :param W: img.W
    :return: index_inside
    """
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


def _unmap(data, count, index, fill=0):
    """
    将在图像内部的框映映射回，并代替原框数据
    """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


####################################################################################
class AnchorTargetCreator(object):
    """
    生成训练所需要的anchor(正负各128个框的坐标和对应label(0 or 1))
     - 对于每一个 GT bbox，选择和它交并比最大的一个 Anchor 作为正样本。
     - 对于剩下的 Anchor，从中选择和任意一个 GT bbox 交并比超过 0.7 的 Anchor 作为正样本，
       正样本数目不超过 128 个。
     - 随机选择和 GT bbox 交并比小于 0.3 的 Anchor 作为负样本，负样本和正样本的总数为256

    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size
        n_anchor = len(anchor)
        # 将那些超出图片范围的anchor全部去掉,只保留位于图片内部的序号
        inside_index = _get_inside_index(anchor, img_H, img_W)
        # 保留位于图片内部的anchor
        anchor = anchor[inside_index]
        # 筛选出符合条件的正例128个负例128并给它们附上相应的label
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # 计算每一个anchor与对应bbox求得iou最大的bbox计算偏移
        # 量（注意这里是位于图片内部的每一个）
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # 将位于图片内部的框的label对应到所有生成的20000个框中
        # （label原本为所有在图片中的框的）
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        # 将回归框对应到所有生成的20000个框中（label原的本为
        # 所有在图片中的框的）
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, index, anchor, bbox):
        label = np.empty((len(index),), dtype=np.int32)
        # 全部填充-1
        label.fill(-1)
        # 调用_calc_ious（）函数得到每个anchor与哪个bbox的iou最大
        # 以及这个iou值、每个bbox与哪个anchor的iou最大(需要体会从
        # 行和列取最大值的区别)
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, index)
        # 把每个anchor与对应的框求得的iou值与负样本阈值比较，若小
        # 于负样本阈值，则label设为0，pos_iou_thresh=0.7,
        # neg_iou_thresh=0.3
        label[max_ious < self.neg_iou_thresh] = 0

        # 把与每个bbox求得iou值最大的anchor的label设为1
        label[gt_argmax_ious] = 1

        # 把每个anchor与对应的框求得的iou值与正样本阈值比较，
        # 若大于正样本阈值，则label设为1
        label[max_ious >= self.pos_iou_thresh] = 1

        # 按照比例计算出正样本数量，pos_ratio=0.5，n_sample=256
        n_pos = int(self.pos_ratio * self.n_sample)
        # 得到所有正样本的索引,行，列
        pos_index = np.where(label == 1)[0]

        # 如果选取出来的正样本数多于预设定的正样本数，则随机抛弃,并置为-1
        if len(pos_index) > n_pos:
            disable_index1 = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index1] = -1

        # 设定的负样本的数量
        n_neg = self.n_sample - np.sum(label == 1)
        # 负样本的索引
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            # 随机选择不要的负样本，个数为len(neg_index)-neg_index，label值设为-1
            disable_index2 = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index2] = -1

        return argmax_ious, label

    ####################################################################################
    def _calc_ious(self, anchor, bbox, index):
        # 调用bbox_iou函数计算anchor与bbox的IOU， ious：（N,K），
        # N为anchor中第N个，K为bbox中第K个，N大概有15000个
        ious = bbox_iou(anchor, bbox)
        # 1代表行，0代表列
        argmax_ious = ious.argmax(axis=1)
        # 求出每个anchor与哪个bbox的iou最大，以及最大值，max_ious:[1,N]
        max_ious = ious[np.arange(len(index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        # 求出每个bbox与哪个anchor的iou最大，以及最大值,gt_max_ious:[1,K]
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        # 然后返回最大iou的索引（每个bbox与哪个anchor的iou最大),有K个
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


class ProposalCreator:
    """
     通过2W+ 的anchor得到目标框bboxe->roi，利用最小尺寸过滤roi
     然后通过排序,从中选取概率较大的12000张，
     利用非极大值抑制，选出2000个ROIS位置参数.
     最终得到经过筛选的2K or 300个bbox(roi)
    :param 给anchor、位置偏差loc、anchor与框的score
    """

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
        # 这里的loc和score是经过region_proposal_network中
        # 经过1x1卷积分类和回归得到的.

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms  # 12000
            n_post_nms = self.n_train_post_nms  # 经过NMS后有2000个
        else:
            n_pre_nms = self.n_test_pre_nms  # 6000
            n_post_nms = self.n_test_post_nms  # 经过NMS后有300个

        # 将bbox转换为近似groudtruth的anchor(即rois)
        roi = loc2bbox(anchor, loc)
        # slice表示切片操作(start，stop，step)->0，2|1，3
        # 裁剪将rois的ymin,ymax限定在[0,H],小于0的都为0，大于H的都为H
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        # 裁剪将rois的xmin,xmax限定在[0,W]
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale  # 16
        # rois's width
        hs = roi[:, 2] - roi[:, 0]
        # rois's height
        ws = roi[:, 3] - roi[:, 1]
        # 确保rois的长宽大于最小阈值 #############################################################
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        # 对剩下的ROIs进行打分（根据region_proposal_network中rois的预测前景概率)
        score = score[keep]
        # 将score拉伸并逆序（从高到低）排序
        order = score.ravel().argsort()[::-1]
        # train时从20000中取前12000个rois，test取前6000个
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # （具体需要看NMS的原理以及输入参数的作用）调用非极大值抑制函数，
        # 将重复的抑制掉，就可以将筛选后ROIS进行返回。经过NMS处理后Train
        # 数据集得到2000个框，Test数据集得到300个框
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi


class ProposalTargetCreator(object):
    """
    - ProposalCreator产生2000个ROIS，但是这些ROIS并不都用于训练，
    - 经过本ProposalTargetCreator的筛选产生128个用于自身的训练
    """

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio  # 选前景的比例
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    # 输入：2000个rois、一个batch（一张图）中所有的bbox ground truth（R，4）、
    # 对应bbox所包含的label（R，1）
    # 输出：128个sample roi（128，4）、128个gt_roi_loc（128，4）、
    # 128个gt_roi_label（128，1）
    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape
        # 首先将2000个roi和m个bbox给concatenate了一下成为
        # 新的roi（2000+m，4）。
        roi = np.concatenate((roi, bbox), axis=0)
        # n_sample = 128,pos_ratio=0.25，round 对传入的数据进行四舍五入
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        # 计算每一个roi与每一个bbox的iou
        iou = bbox_iou(roi, bbox)
        # 按行找到最大值，返回最大值对应的序号以及其真正的IOU。
        # 返回的是每个roi与哪个bbox的最大，以及最大的iou值
        gt_assignment = iou.argmax(axis=1)
        # 每个roi与对应bbox最大的iou
        max_iou = iou.max(axis=1)
        # 从1开始的类别序号，给每个类得到真正的label(将0-1变为1-2)
        gt_roi_label = label[gt_assignment] + 1
        # 同样的根据iou的最大值将正负样本找出来，pos_iou_thresh=0.5
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        # 需要保留的roi个数（满足大于pos_iou_thresh条件的roi与64之间较小的一个）
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        # 找出的样本数目过多就随机丢掉一些
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)
        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # 此时输出的128*4的sample_roi就可以去扔到 RoIHead网络里去进行分类
        # 与回归了。同样， RoIHead网络利用这sample_roi+featue为输入，输出
        # 是分类（1类）和回归（进一步微调bbox）的预测值，那么分类回归的groud
        # truth就是ProposalTargetCreator输出的gt_roi_label和gt_roi_loc。
        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label


# 3.model
def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


class RegionProposalNetwork(torch.nn.Module):
    """
    图片网络框架，除backbone
    """

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        # 首先生成上述以（0，0）为中心的9个base anchor
        self.anchor_base = one_point_generate_anchor(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        # 2W+ 的anchor得到2K的目标框bboxe->roi
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = torch.nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = torch.nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """
        :param x: 从图像中提取的特征。它的形状是:math:`(N, C, H, W)`
        :param img_size: 一个元组：obj：`height，width`，其中包含缩放后的图像大小
        :param scale: 对输入图像进行的缩放量,从文件中读取它们
        :return:
        """
        n, _, hh, ww = x.shape
        # 在9个base_anchor基础上生成hh*ww*9个anchor，对应到原图坐标
        # feat_stride=16 ，因为是经4次pool后提到的特征，故feature map较
        # 原图缩小了16倍
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)
        # （hh * ww * 9）/hh*ww = 9
        n_anchor = anchor.shape[0] // (hh * ww)
        # 512个3x3卷积(512, H/16,W/16)
        h = F.relu(self.conv1(x))
        # n_anchor（9）* 4个1x1卷积，回归坐标偏移量。（9*4，hh,ww)
        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
        # 转换为（n，hh，ww，9*4）后变为（n，hh*ww*9，4）
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # ---------------------------
        # n_anchor（9）*2个1x1卷积，回归类别。（9*2，hh,ww）
        rpn_scores = self.score(h)
        # 转换为（n，hh，ww，9 * 2）
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        # 计算{Softmax}(x_{i}) = \{exp(x_i)}{\sum_j exp(x_j)}
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        # 得到前景的分类概率
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        # 得到所有anchor的前景分类概率
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        # 得到每一张feature map上所有anchor的网络输出值
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        # n为batch_size数
        for i in range(n):
            # 调用ProposalCreator函数， rpn_locs维度（hh*ww*9，4）
            # ，rpn_fg_scores维度为（hh*ww*9），anchor的维度为
            # （hh*ww*9，4）， img_size的维度为（3，H，W），H和W是
            # 经过数据预处理后的。计算（H/16）x(W/16)x9(大概20000)
            # 个anchor属于前景的概率，取前12000个并经过NMS得到2000个
            # 近似目标框G^的坐标。roi的维度为(2000,4)
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            # rois为所有batch_size的roi
            rois.append(roi)
            roi_indices.append(batch_index)
        # 按行拼接（即没有batch_size的区分，每一个[]里都是一个anchor的四个坐标）
        rois = np.concatenate(rois, axis=0)
        # 一个batch只会输入一张图象。如果多张图像的话就需要存储索引以找到对应图像的roi
        roi_indices = np.concatenate(roi_indices, axis=0)
        # rpn_locs的维度（hh*ww*9，4），rpn_scores维度为（hh*ww*9，2），
        # rois的维度为（2000,4），roi_indices用不到，
        # anchor的维度为（hh*ww*9，4）
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)

    return new_f


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch.Tensor):
        return data.item()


class FasterRCNN(torch.nn.Module):
    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
                 ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        # Total number              of classes including the background.
        return self.head.n_class

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(
            h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    # predict函数是对网络RoIhead网络输出的预处理
    # 函数_suppress将得到真正的预测结果。
    # 此函数是一个按类别的循环，l从1至n（0类为背景类）(本次为1到1)。
    # 即预测思想是按n个类别顺序依次验证，如果有满足该类的预测结果，
    # 则记录，否则转入下一类（一张图中也就几个类别而已）。例如筛选
    # 预测出第1类的结果，首先在cls_bbox中将所有128个预测第1类的
    # bbox坐标找出，然后从prob中找出128个第1类的概率。因为阈值为0.7，
    # 也即概率>0.7的所有边框初步被判定预测正确，记录下来。然而可能有
    # 多个边框预测第1类中同一个物体，同类中一个物体只需一个边框，
    # 所以需再经基于类的NMS后使得每类每个物体只有一个边框，至此
    # 第1类预测完成，记录第1类的所有边框坐标、标签、置信度。
    # 接着下一类...，直至n类都记录下来，那么一张图片（也即一个batch）
    # 的预测也就结束了。
    def _suppress(self, raw_cls_bbox, raw_prob):
        """
        得到真正的预测结果.
        :param raw_cls_bbox:
        :param raw_prob:
        :return:
        """
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)
            # import ipdb;ipdb.set_trace()
            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @nograd
    def predict(self, imgs, sizes=None, visualize=False):
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = totensor(img[None]).float()
            # 对读入的图片求尺度scale，因为输入的图像经预处理就会有缩放，
            # 所以需记录缩放因子scale，这个缩放因子在ProposalCreator
            # 筛选roi时有用到，即将所有候选框按这个缩放因子映射回原图，
            # 超出原图边框的趋于将被截断.
            scale = img.shape[3] / size[1]
            # 执行forward
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            # 为ProposalCreator对loc做了归一化（-mean /std）处理，所以这里
            # 需要再*std+mean，此时的位置参数loc为roi_cls_loc。然后将这128
            # 个roi利用roi_cls_loc进行微调，得到新的cls_bbox。
            mean = torch.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(tonumpy(roi).reshape((-1, 4)),
                                tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])
            # 对于分类得分roi_scores，我们需要将其经过softmax后转为概率prob。
            # 值得注意的是我们此时得到的是对所有输入128个roi以及位置参数、得分
            # 的预处理，下面将筛选出最后最终的预测结果.
            prob = (F.softmax(totensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()  # 这次预测之后下次要接着训练
        return bboxes, labels, scores

    # 定义了优化器optimizer，对于需要求导的参数 按照是否含bias赋予不同的学习率。
    # 默认是使用SGD，可选Adam，不过需更小的学习率
    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer


def decom_vgg16():
    if opt.vgg_pretrain:
        model = vgg16(pretrained=False)
        if not opt.pretrain_path:
            # 加载参数信息
            model.load_state_dict(torch.load(opt.vgg_pretrain_path))
    else:
        model = vgg16(not opt.pretrain_path)
    # 加载预训练模型vgg16的conv5_3之前的部分
    features = list(model.features)[:30]
    classifier = model.classifier
    # 分类部分放到一个list里面
    classifier = list(classifier)
    # 删除输出分类结果层
    del classifier[6]
    if not opt.use_drop:
        # 删除两个dropout
        del classifier[5]
        del classifier[2]
    classifier = torch.nn.Sequential(*classifier)
    # freeze top4 conv
    # 冻结vgg16前2个stage,不进行反向传播
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    # 拆分为特征提取网络和分类网络
    return torch.nn.Sequential(*features), classifier

    # 分别对特征VGG16的特征提取部分、分类部分、RPN网络、
    # VGG16RoIHead网络进行了实例化


class VGG16RoIHead(torch.nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()
        # vgg16中的最后两个全连接层
        self.classifier = classifier
        self.cls_loc = torch.nn.Linear(4096, n_class * 4)
        self.score = torch.nn.Linear(4096, n_class)
        # 全连接层权重初始化
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        # 加上背景2类
        self.n_class = n_class
        # 7x7
        self.roi_size = roi_size
        # 1/16
        self.spatial_scale = spatial_scale
        # 将大小不同的roi变成大小一致，得到pooling后的特征，
        # 大小为[300, 512, 7, 7]。利用Cupy实现在线编译的
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        # 前面解释过这里的roi_indices其实是多余的，因为batch_size一直为1
        # in case roi_indices is  ndarray
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        # 把tensor变成在内存中连续分布的形式
        indices_and_rois = xy_indices_and_rois.contiguous()

        # 接下来分析roi_module.py中的RoI
        pool = self.roi(x, indices_and_rois)
        # flat操作
        pool = pool.view(pool.size(0), -1)
        # decom_vgg16（）得到的calssifier,得到4096
        fc7 = self.classifier(pool)
        # （4096->84）
        roi_cls_locs = self.cls_loc(fc7)
        # （4096->21）
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


class FasterRCNNVGG16(FasterRCNN):
    # vgg16通过5个stage下采样16倍
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    # 总类别数为1类，三种尺度三种比例的anchor
    def __init__(self,
                 n_fg_class=1,
                 ratios=[0.5, 1, 2],
                 anchor_scale=[8, 16, 32]):
        extractor, classifier = decom_vgg16()
        # 返回rpn_locs, rpn_scores, rois, roi_indices, anchor
        rpn = RegionProposalNetwork(512, 512, ratios=ratios,
                                    anchor_scales=anchor_scale,
                                    feat_stride=self.feat_stride, )

        head = VGG16RoIHead(n_class=n_fg_class + 1,
                            roi_size=7,
                            spatial_scale=(1. / self.feat_stride),
                            classifier=classifier)

        super(FasterRCNNVGG16, self).__init__(
            extractor, rpn, head,
        )


# 4.Train
LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss


class FasterRCNNTrainer(torch.nn.Module):
    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        # 下面2个参数是在_faster_rcnn_loc_loss调用用来计算位置损失函数用到的超参数
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets.
        # 用于从20000个候选anchor中产生256个anchor进行二分类和位置回归，也就是
        # 为rpn网络产生的预测位置和预测类别提供真正的ground_truth标准
        self.anchor_target_creator = AnchorTargetCreator()
        # AnchorTargetCreator和ProposalTargetCreator是为了生成训练的目标
        # （或称ground truth），只在训练阶段用到，ProposalCreator是RPN为Fast
        #  R-CNN生成RoIs，在训练和测试阶段都会用到。所以测试阶段直接输进来300
        # 个RoIs，而训练阶段会有AnchorTargetCreator的再次干预
        self.proposal_target_creator = ProposalTargetCreator()
        # (0., 0., 0., 0.)
        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        # (0.1, 0.1, 0.2, 0.2)
        self.loc_normalize_std = faster_rcnn.loc_normalize_std
        # SGD
        self.optimizer = self.faster_rcnn.get_optimizer()

        # 混淆矩阵，就是验证预测值与真实值精确度的矩阵ConfusionMeter
        # (2)括号里的参数指的是类别数
        self.rpn_cm = ConfusionMeter(2)
        # roi的类别有2种（1个object类+1个background）
        self.roi_cm = ConfusionMeter(2)
        # 平均损失
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        # 获取batch个数
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')
        # （n,c,hh,ww）
        _, _, H, W = imgs.shape
        img_size = (H, W)
        # vgg16 conv5_3之前的部分提取图片的特征
        features = self.faster_rcnn.extractor(imgs)
        # rpn_locs的维度（hh*ww*9，4），rpn_scores维度为（hh*ww*9，2），
        #  rois的维度为（2000,4），roi_indices用不到，anchor的维度为
        # （hh*ww*9，4），H和W是经过数据预处理后的。计算（H/16）x(W/16)x9
        # (大概20000)个anchor属于前景的概率，取前12000个并经过NMS得到2000个
        # 近似目标框G^的坐标。roi的维度为(2000,4)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        # bbox维度(N, R, 4)
        bbox = bboxes[0]
        # labels维度为（N，R）
        label = labels[0]
        # hh*ww*9
        rpn_score = rpn_scores[0]
        # hh*ww*9
        rpn_loc = rpn_locs[0]
        # (2000,4)
        roi = rois

        # Sample RoIs and forward
        # 调用proposal_target_creator函数生成sample roi（128，4）、
        # gt_roi_loc（128，4）、gt_roi_label（128，1），RoIHead网络
        # 利用这sample_roi+featue为输入，输出是分类（21类）和回归
        # （进一步微调bbox）的预测值，那么分类回归的groud truth就
        # 是ProposalTargetCreator输出的gt_roi_label和gt_roi_loc。
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            tonumpy(bbox),
            tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = torch.zeros(len(sample_roi))
        # roi回归输出的是128*84和128*21，然而真实位置参数是128*4和真实标签128*1
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        # 输入20000个anchor和bbox，调用anchor_target_creator函数得到
        # 2000个anchor与bbox的偏移量与label
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = totensor(gt_rpn_label).long()
        gt_rpn_loc = totensor(gt_rpn_loc)
        # 下面分析_fast_rcnn_loc_loss函数。rpn_loc为rpn网络回归出来的偏移量
        # （20000个），gt_rpn_loc为anchor_target_creator函数得到2000个anchor
        # 与bbox的偏移量，rpn_sigma=1.
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        # rpn_score为rpn网络得到的（20000个）与anchor_target_creator
        # 得到的2000个label求交叉熵损失
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = tonumpy(rpn_score)[tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        # roi_cls_loc为VGG16RoIHead的输出（128*84）， n_sample=128
        n_sample = roi_cls_loc.shape[0]
        # roi_cls_loc=（128,2,4）
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                              totensor(gt_roi_label).long()]
        # proposal_target_creator()生成的128个proposal与bbox求得的偏移量
        # dx,dy,dw,dh
        gt_roi_label = totensor(gt_roi_label).long()
        # 128个标签
        gt_roi_loc = totensor(gt_roi_loc)
        # 采用smooth_l1_loss
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)
        # 求交叉熵损失
        roi_cls_loss = torch.nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(totensor(roi_score, False), gt_roi_label.data.long())
        # 四个loss加起来
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)

        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        cell = {k: v.value()[0] for k, v in self.meters.items()}
        str_ = {k: str(v.value()[0])[:6] for k, v in self.meters.items()}
        return cell, str_


# AP和mAP URL:https://www.cnblogs.com/ginkgo-/p/13737771.html
def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        iou_thresh=0.5, use_07_metric=False):
    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        iou_thresh=0.5, gt_difficults=None):
    # 将于图片有关的变量转换为迭代器，每次迭代的作用针对下一张图片进行检测
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    # 因为VOC数据集中存在难以识别的Ground Truth，所以加上difficult，
    # 如果是difficult可以不参与计算，在下面的代码中体现为标记为-1
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    # 初始化记录Precision和Recall数量的字典，第一级键应该是类别
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    # 对每张图片进行运算，统计各类别的匹配和不匹配的数量
    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels):
        gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            # 获取l类的pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]
            # 将预测类别为l的pred_box和pred_score收集起来并且降序排列
            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            # 表示一共有多少个ground truth，这里就可以发现忽视了难以判别的ground truth
            score[l].extend(pred_score_l)
            if len(pred_bbox_l) == 0:  # 如果这个图片l类的bbox不存在
                continue
            if len(gt_bbox_l) == 0:  # 如果这个图片存在l类的bbox，但是没有l类的ground truth 就可以标记未不匹配
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC评估遵循整数类型的边界框
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            # 计算这张图片l类的bbox和ground truth 之间的IoU，生成的矩阵大小为(M,N)
            # M :pred_bbox_l的长度，N:gt_bbox_l的长度
            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            # 这里应该是找到和pred_bbox_l中IOU最大的ground truth，shape:(M,)
            # pred_bbox_l和ground truth最大的iou都小于阈值的标记为不匹配的，-1
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:  # 该位置的pred_bbox存在超过IOU阈值的ground truth，就标记为匹配成功，1
                        match[l].append(1)
                    else:  # 一个ground truth只能被对应一次
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
            # match应该表示pred_bbox有多少个对应上了ground truth，而且留下了列表，可以记录预测框的对应情况

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')
    # 保证相同长度

    n_fg_class = max(n_pos.keys()) + 1  # 总共的类别数量
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)
        # 获取按照得分降序排列的所有pred_bbox
        order = score_l.argsort()[::-1]
        # 匹配列表也按照得分降序排列
        match_l = match_l[order]
        # 这里的指标是AP
        # 这里按照得分降序排列计算出tp和fp，是为了计算PR曲线和AP。
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # 如果fp+tp == 0 ,那么prec[l] = nan
        prec[l] = tp / (fp + tp)
        # 如果n_pos[l] <= 0,那么rec[l] = None
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    # 初始化记录的变量
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        # 提前做成判断，防止程序报错
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:  # 是否使用VOC2007计算方法
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:  # VOC 2007以后的方法
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def plot_many(d):
    """
    plot multi values
    @params d: dict (name,value) i.e. ('loss',0.11)
    """
    for k, v in d.items():
        if v is not None:
            plt.plot(k, v)


def inverse_normalize(img):
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def vis_image(img, ax=None):
    """Visualize a color image.
    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax


def vis_bbox(img, bbox, label=None, score=None, ax=None):
    """Visualize bounding boxes inside image.
    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    """

    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax


def fig2data(fig):
    """
    brief Convert a Matplotlib figure to a 4D numpy array with RGBA
    channels and return it
    @param fig: a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig4vis(fig):
    """
    convert figure to ndarray
    """
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.int32)
    plt.close()
    # HWC->CHW
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.


def visdom_bbox(*args, **kwargs):
    fig = vis_bbox(*args, **kwargs)
    data = fig4vis(fig)
    return data


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  )
    testset = TestDataset(opt)
    # 将数据装载到dataloader中，shuffle=True允许数据打乱排序，
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       shuffle=False,
                                       pin_memory=True
                                       )
    # 定义faster_rcnn=FasterRCNNVGG16()训练模型
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    # 设置trainer = FasterRCNNTrainer(faster_rcnn).cuda()将
    # FasterRCNNVGG16作为fasterrcnn的模型送入到FasterRCNNTrainer
    # 中并设置好GPU加速
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.pretrain_path:
        trainer.load(opt.pretrain_path)
        print('load pretrained model from %s' % opt.pretrain_path)

    best_map = 0
    T_map = False
    lr_ = opt.lr
    plt_Map = []
    plt_total_loss = []
    plt_rpn_loc_loss = []
    plt_rpn_cls_loss = []
    plt_roi_loc_loss = []
    plt_roi_cls_loss = []
    # 用一个for循环开始训练过程，而训练迭代的次数
    # opt.epoch=10也在config.py文件中预先定义好，属于超参数
    for epoch in range(opt.epoch):
        print(f'epoch:{epoch + 1}\n')
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = scalar(scale)
            # 然后从训练数据中枚举dataloader,设置好缩放范围，
            # 将img,bbox,label,scale全部设置为可gpu加速
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # (img,bbox,label,scale)进行一次参数迭代优化过程
            trainer.train_step(img, bbox, label, scale)
            if (ii + 1) % opt.plt_train == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                # plot loss
                # plot_many(trainer.get_meter_data()[0])

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     tonumpy(bbox_[0]),
                                     tonumpy(label_[0]))
                # plt.subplt() 画原图与GT

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       tonumpy(_bboxes[0]),
                                       tonumpy(_labels[0]).reshape(-1),
                                       tonumpy(_scores[0]))
                # plt.subplt() 画原图与pre
                # rpn confusion matrix(meter)
                # trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                # trainer.vis.img('roi_cm', totensor(trainer.roi_cm.conf, False).float())
                # plt.cla()
                # plt.subplot(1, 2, 1)
                # plt.imshow(gt_img.transpose(1, 2, 0))
                # plt.subplot(1, 2, 2)
                # plt.imshow(pred_img.transpose(1, 2, 0))
                # plt.title('train')
                # plt.show()

        # 调用eval函数计算map等指标
        eval_result = eval(test_dataloader, faster_rcnn, epoch + 1)
        # 可视化map
        #  trainer.vis.plot('test_map', eval_result['map'])
        # 设置学习的learning rate
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},\nloss:{}'.format(str(lr_),
                                                    str(eval_result),  # ['map']
                                                    trainer.get_meter_data()[1])
        # 将损失学习率以及map等信息及时显示更新
        print(log_info)
        plt_Map.append(eval_result['map'])
        plt_total_loss.append(trainer.get_meter_data()[0]['total_loss'])
        plt_rpn_loc_loss.append(trainer.get_meter_data()[0]['rpn_loc_loss'])
        plt_rpn_cls_loss.append(trainer.get_meter_data()[0]['rpn_cls_loss'])
        plt_roi_loc_loss.append(trainer.get_meter_data()[0]['roi_loc_loss'])
        plt_roi_cls_loss.append(trainer.get_meter_data()[0]['roi_cls_loss'])
        # trainer.vis.log(log_info)
        # 用if判断语句永远保存效果最好的map
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
            T_map = True
        if (epoch + 1) % opt.plt_test == 0:
            # if判断语句如果学习的epoch达到了9就将学习率*0.1
            # 变成原来的十分之一
            if T_map:
                trainer.load(best_path)
                T_map = False
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

            plt.plot(plt_Map, '-', label="map")
            plt.plot(plt_total_loss, '-', label="total_loss")
            plt.plot(plt_rpn_loc_loss, ":", label="rpn_loc_loss")
            plt.plot(plt_rpn_cls_loss, ":", label="rpn_cls_loss")
            plt.plot(plt_roi_loc_loss, '--', label="roi_loc_loss")
            plt.plot(plt_roi_cls_loss, '--', label="roi_cls_loss")
            plt.title('Loss-' + str(epoch+1))
            plt.legend()
            plt.show()


def eval(dataloader, faster_rcnn, epoch):
    plt_i = random.randint(0, len(dataloader))
    # 预测框的位置，预测框的类别和分数
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    # 真实框的位置，类别，是否为明显目标
    gt_bboxes, gt_labels = list(), list()
    # 一个for循环，从 enumerate(dataloader)里面依次读取数据，
    # 读取的内容是: imgs图片，sizes尺寸，gt_boxes真实框的位置
    #  gt_labels真实框的类别
    for ii, (imgs, gt_img, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        # 用faster_rcnn.predict(imgs,[sizes]) 得出预测的pred_boxes_,
        # pred_labels_,pred_scores_预测框位置，预测框标记以及预测框的分数等等
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        # if ii == test_num: break
        # 将pred_bbox,pred_label,pred_score ,gt_bbox,gt_label
        # 预测和真实的值全部依次添加到开始定义好的列表里面去，如果迭代次数等于测
        # 试test_num，那么就跳出循环！调用 eval_detection_voc函数，接收上述的
        # 六个列表参数，完成预测水平的评估！得到预测的结果
        if ii == plt_i:
            images = gt_img[0].permute(1, 2, 0).numpy().astype(np.uint8)
            temp_plt = plt.imshow(images)
            for y1, x1, y2, x2 in pred_bboxes_[0].reshape(-1, 4):
                temp_plt.axes.add_patch(
                    patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
            for y1, x1, y2, x2 in gt_bboxes_.numpy().reshape(-1, 4):
                temp_plt.axes.add_patch(
                    patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='yellow', linewidth=2))
            plt.title('Epoch:' + str(epoch) + '_TestImg')
            plt.show()
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        use_07_metric=False)
    return result


if __name__ == '__main__':
    train()

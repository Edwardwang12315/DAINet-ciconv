#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from setuptools.sandbox import save_path
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models.factory import build_net
from torchvision.utils import make_grid
import glob

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import csv  # 新增导入csv模块

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benckmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def tensor_to_image(tensor):
    grid = make_grid(tensor)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr

def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image

def detect_face(img, tmp_shrink):
    image = cv2.resize(img, None, None, fx=tmp_shrink,
                       fy=tmp_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x = x / 255.
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    
    if use_cuda:
        x = x.cuda()

    y = net.test_forward(x)[0]
    detections = y.data.cpu().numpy()

    # # 转换为 0~255 的 uint8 类型
    # image = (image * 255).astype( np.uint8 )
    #
    # # 显示图像
    # plt.imshow( image )
    # plt.axis( 'off' )
    # plt.show()
    # exit()

    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    boxes=[]
    scores = []
    for i in range(detections.shape[1]):
      j = 0
      while ((j < detections.shape[2]) and detections[0, i, j, 0] > 0.0):
        pt = (detections[0, i, j, 1:] * scale)
        score = detections[0, i, j, 0]
        boxes.append([pt[0],pt[1],pt[2],pt[3]])
        scores.append(score)
        j += 1

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det_xmin = boxes[:,0] # / tmp_shrink
    det_ymin = boxes[:,1] # / tmp_shrink
    det_xmax = boxes[:,2] # / tmp_shrink
    det_ymax = boxes[:,3] # / tmp_shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    return det


def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s,detect_face(image, 0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b,detect_face(image, 1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink: # and bt <= 2:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2

        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def multi_scale_test_pyramid(image, max_shrink):
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


def bbox_vote(det_):
    order_ = det_[:, 4].ravel().argsort()[::-1]
    det_ = det_[order_, :]
    dets_ = np.zeros((0, 5),dtype=np.float32)
    while det_.shape[0] > 0:
        # IOU
        area_ = (det_[:, 2] - det_[:, 0] + 1) * (det_[:, 3] - det_[:, 1] + 1)
        xx1 = np.maximum(det_[0, 0], det_[:, 0])
        yy1 = np.maximum(det_[0, 1], det_[:, 1])
        xx2 = np.minimum(det_[0, 2], det_[:, 2])
        yy2 = np.minimum(det_[0, 3], det_[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o_ = inter / (area_[0] + area_[:] - inter)

        # get needed merge det and delete these det
        merge_index_ = np.where(o_ >= 0.3)[0]
        det_accu_ = det_[merge_index_, :]
        det_ = np.delete(det_, merge_index_, 0)

        if merge_index_.shape[0] <= 1:
            continue
        det_accu_[:, 0:4] = det_accu_[:, 0:4] * np.tile(det_accu_[:, -1:], (1, 4))
        max_score_ = np.max(det_accu_[:, 4])
        det_accu_sum_ = np.zeros((1, 5))
        det_accu_sum_[:, 0:4] = np.sum(det_accu_[:, 0:4], axis=0) / np.sum(det_accu_[:, -1:])
        det_accu_sum_[:, 4] = max_score_
        try:
            dets_ = np.row_stack((dets_, det_accu_sum_))
        except:
            dets_ = det_accu_sum_

    dets_ = dets_[0:750, :]
    return dets_


def load_models():
    print('build network')
    net = build_net('test', num_classes=2, model='dark')
    net.eval()
    net.load_state_dict(torch.load('../../model/forDAINet/dark/dsfd_v1.5.pth')) # Set the dir of your model weight

    if use_cuda:
        net = net.cuda()

    return net

def draw_boxes_with_matplotlib(image, dets,save_path):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.axis('off')

    for det in dets:
        xmin, ymin, xmax, ymax, score = det
        if score>0.8:
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, f'{score:.2f}', color='r', fontsize=6)

    plt.savefig( save_path , bbox_inches = 'tight' ,dpi = 600 ,pad_inches=0)  # 保存为高分辨率图片
    # plt.show(block=False)# 控制是否停留

# 新增计算IoU的函数
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = inter_xmax - inter_xmin
    inter_height = inter_ymax - inter_ymin

    if inter_width <=0 or inter_height <=0:
        return 0.0

    area_inter = inter_width * inter_height
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    iou = area_inter / (area1 + area2 - area_inter + 1e-10)
    return iou

# 新增计算mAP的函数
def compute_mAP(detections_path, ground_truth_path, iou_threshold=0.5):
    ground_truths = {}
    # 读取真实标签
    for gt_file in glob.glob(os.path.join(ground_truth_path, '*.txt')):
        image_id = os.path.splitext(os.path.basename(gt_file))[0]
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        # 跳过首行（边界框数量）
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            try:
                xmin = float(parts[0])
                ymin = float(parts[1])
                xmax = float(parts[2])
                ymax = float(parts[3])
                boxes.append({'class': 0, 'bbox': [xmin, ymin, xmax, ymax]})
            except ValueError:
                continue
        
        ground_truths[image_id] = boxes

    # 读取检测结果
    detections = {}
    for det_file in glob.glob(os.path.join(detections_path, '*.txt')):
        image_id = os.path.splitext(os.path.basename(det_file))[0]
        with open(det_file, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                xmin = float(parts[0])
                ymin = float(parts[1])
                xmax = float(parts[2])
                ymax = float(parts[3])
                score = float(parts[4])
                boxes.append({'bbox': [xmin, ymin, xmax, ymax], 'score': score})
            except ValueError:
                continue
        
        boxes.sort(key=lambda x: x['score'], reverse=True)
        detections[image_id] = boxes

    # 计算AP
    class_id = 0
    tp_list = []
    fp_list = []
    scores_list = []
    n_gt = 0

    for image_id in ground_truths:
        gt_boxes = [box for box in ground_truths[image_id] if box['class'] == class_id]
        n_gt += len(gt_boxes)
        det_boxes = detections.get(image_id, [])
        
        used_gt = set()
        for det in det_boxes:
            scores_list.append(det['score'])
            max_iou = 0.0
            matched_gt_idx = -1
            
            for idx, gt in enumerate(gt_boxes):
                iou = calculate_iou(det['bbox'], gt['bbox'])
                if iou > max_iou and idx not in used_gt:
                    max_iou = iou
                    matched_gt_idx = idx
            
            if max_iou >= iou_threshold:
                tp_list.append(1)
                fp_list.append(0)
                used_gt.add(matched_gt_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

    if n_gt == 0:
        return 0.0

    # 排序并计算precision/recall
    sorted_indices = np.argsort(-np.array(scores_list))
    tp = np.array(tp_list)[sorted_indices]
    fp = np.array(fp_list)[sorted_indices]

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / n_gt

    # 计算AP
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if np.any(mask):
            ap += np.max(precision[mask]) / 11.0
        else:
            ap += 0.0

    return ap

if __name__ == '__main__':

    ''' Parameters '''

    USE_MULTI_SCALE = True
    MY_SHRINK = 1

    # USE_MULTI_SCALE = False
    # MY_SHRINK = 2

    save_path = './result'

    def load_images():
      imglist = glob.glob('../../dataset/DarkFace/image/*.png') # Set the dir of your test data
      return imglist

    ''' Main Test '''

    net = load_models()
    img_list = load_images()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    now = 0
    print('Processing: {}/{}'.format(now+1, img_list.__len__()))
    for img_path in img_list:
        # Load images       
        image = Image.open(img_path)
        # if image.mode == 'L':
        #     image = image.convert('RGB')
        image = np.array(image)

        # Face Detection
        max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5 # the max size of input image for caffe
        max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink

        if USE_MULTI_SCALE:
            with torch.no_grad():
                det0 = detect_face(image, MY_SHRINK)  # origin test
                det1 = flip_test(image, MY_SHRINK)    # flip test
                [det2, det3] = multi_scale_test(image, max_im_shrink) # multi-scale test
                det4 = multi_scale_test_pyramid(image, max_im_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)
        else:
            with torch.no_grad():
                dets = detect_face(image, MY_SHRINK)  # origin test

        # Save result
        fout = open(os.path.join(save_path,"annotations", Path(os.path.basename(img_path)).stem + '.txt'), 'w')

        for i in range(dets.shape[0]):
            xmin = dets[i][0]
            ymin = dets[i][1]
            xmax = dets[i][2]
            ymax = dets[i][3]
            score = dets[i][4]
            fout.write('{} {} {} {} {}\n'.format(xmin, ymin, xmax, ymax, score))
        now += 1
        print('Processing: {}/{}'.format(now + 1, img_list.__len__()))

        # 在代码中调用绘制函数
        image = Image.open( img_path )
        if image.mode == 'L' :
            image = image.convert( 'RGB' )
        image = np.array( image )

        # 假设 dets 是检测到的框
        img_save=os.path.join(save_path,"images",Path(os.path.basename(img_path)).stem + '.png')
        draw_boxes_with_matplotlib( image , dets,img_save)
    
    # 统计mAP
    ground_truth_path = '../dataset/DarkFace/label'  # 修改为你的真实标签路径
    detection_path = os.path.join(save_path, 'annotations')
    if os.path.exists(ground_truth_path):
        ap = compute_mAP(detection_path, ground_truth_path)
        print(f'mAP at IoU 0.5: {ap:.4f}')
    else:
        print('真实标签路径不存在，跳过mAP计算')

